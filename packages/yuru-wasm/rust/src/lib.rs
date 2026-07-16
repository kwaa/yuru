#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOCATOR: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    core::arch::wasm32::unreachable()
}

fn integrate_particle(
    next: &mut [f32],
    next_previous: &mut [f32],
    position: &[f32],
    previous: &[f32],
    acceleration: &[f32],
    inverse_mass: f32,
    delta_squared: f32,
    velocity_scale: f32,
) {
    if inverse_mass == 0.0 {
        return;
    }
    for axis in 0..3 {
        let value = position[axis];
        let velocity = (value - previous[axis]) * velocity_scale;
        next_previous[axis] = value;
        next[axis] = value + velocity + acceleration[axis] * delta_squared;
    }
}

fn integrate_serial(
    next: &mut [f32],
    next_previous: &mut [f32],
    positions: &[f32],
    previous: &[f32],
    inverse_masses: &[f32],
    accelerations: &[f32],
    delta_squared: f32,
    velocity_scale: f32,
) {
    for (particle, inverse_mass) in inverse_masses.iter().copied().enumerate() {
        let offset = particle * 3;
        integrate_particle(
            &mut next[offset..offset + 3],
            &mut next_previous[offset..offset + 3],
            &positions[offset..offset + 3],
            &previous[offset..offset + 3],
            &accelerations[offset..offset + 3],
            inverse_mass,
            delta_squared,
            velocity_scale,
        );
    }
}

/// Integrates packed xyz particle state and returns positions followed by the
/// previous-position buffer. The release build enables wasm32 SIMD128 so LLVM
/// can vectorize independent particle batches without changing the wire ABI.
#[wasm_bindgen]
pub fn integrate(
    positions: &[f32],
    previous: &[f32],
    inverse_masses: &[f32],
    accelerations: &[f32],
    delta: f32,
    damping: f32,
) -> Vec<f32> {
    assert_eq!(positions.len(), previous.len());
    assert_eq!(positions.len(), accelerations.len());
    assert_eq!(positions.len(), inverse_masses.len() * 3);

    let mut next = positions.to_vec();
    let mut next_previous = previous.to_vec();
    let delta_squared = delta * delta;
    let velocity_scale = 1.0 - damping.clamp(0.0, 1.0);

    integrate_serial(
        &mut next,
        &mut next_previous,
        positions,
        previous,
        inverse_masses,
        accelerations,
        delta_squared,
        velocity_scale,
    );

    next.extend_from_slice(&next_previous);
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_particles_do_not_move() {
        let result = integrate(
            &[1.0, 2.0, 3.0],
            &[0.0, 1.0, 2.0],
            &[0.0],
            &[0.0, -9.81, 0.0],
            1.0 / 60.0,
            0.0,
        );
        assert_eq!(&result[..3], &[1.0, 2.0, 3.0]);
    }
}
