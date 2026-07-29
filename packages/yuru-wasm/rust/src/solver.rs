use alloc::vec::Vec;
use wasm_bindgen::prelude::*;

const EPSILON: f32 = 1.0e-7;
const STRETCH_EDGE: u8 = 0;

#[cfg(target_arch = "wasm32")]
#[inline]
fn square_root(value: f32) -> f32 {
    use core::arch::wasm32::{f32x4_extract_lane, f32x4_splat, f32x4_sqrt};

    f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(value)))
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn square_root(value: f32) -> f32 {
    value.sqrt()
}

pub(crate) struct Body {
    pub(crate) adjacency: Vec<u32>,
    pub(crate) adjacency_offsets: Vec<u32>,
    pub(crate) collision_cell_size: f32,
    pub(crate) collision_layer: i32,
    pub(crate) collision_layer_axis: [f32; 3],
    pub(crate) has_collision_layer_axis: bool,
    pub(crate) edge_kinds: Vec<u8>,
    pub(crate) edge_particles: Vec<u32>,
    pub(crate) edge_rest_lengths: Vec<f32>,
    pub(crate) filter_group: u32,
    pub(crate) filter_mask: u32,
    pub(crate) indices: Vec<u32>,
    pub(crate) initial: Vec<f32>,
    pub(crate) inverse_masses: Vec<f32>,
    pub(crate) material_drags: Vec<f32>,
    pub(crate) material_frictions: Vec<f32>,
    pub(crate) material_lifts: Vec<f32>,
    pub(crate) material_thicknesses: Vec<f32>,
    pub(crate) maximum_self_collision_depenetration: f32,
    pub(crate) positions: Vec<f32>,
    pub(crate) previous: Vec<f32>,
    pub(crate) self_collision: bool,
    pub(crate) tether_anchors: Vec<u32>,
    pub(crate) tether_lengths: Vec<f32>,
    pub(crate) tether_particles: Vec<u32>,
    pub(crate) triangle_material_indices: Vec<u16>,
    pub(crate) triangle_particles: Vec<u32>,
    pub(crate) triangle_rest_areas: Vec<f32>,
}

impl Body {
    fn material_index(&self, triangle: usize) -> usize {
        self.triangle_material_indices
            .get(triangle)
            .copied()
            .unwrap_or(0) as usize
    }

    pub(crate) fn friction_for_triangle(&self, triangle: usize) -> f32 {
        self.material_frictions
            .get(self.material_index(triangle))
            .copied()
            .unwrap_or_else(|| self.material_frictions[0])
    }

    pub(crate) fn thickness_for_triangle(&self, triangle: usize) -> f32 {
        self.material_thicknesses
            .get(self.material_index(triangle))
            .copied()
            .unwrap_or_else(|| self.material_thicknesses[0])
    }

    fn integrate(&mut self, accelerations: &[f32], delta: f32, damping: f32) {
        let delta_squared = delta * delta;
        let velocity_scale = 1.0 - damping.clamp(0.0, 1.0);
        for (particle, inverse_mass) in self.inverse_masses.iter().copied().enumerate() {
            if inverse_mass == 0.0 {
                continue;
            }
            let offset = particle * 3;
            for axis in 0..3 {
                let index = offset + axis;
                let value = self.positions[index];
                let velocity = (value - self.previous[index]) * velocity_scale;
                self.previous[index] = value;
                self.positions[index] = value + velocity + accelerations[index] * delta_squared;
            }
        }
    }

    fn apply_aerodynamics(&mut self, delta: f32, wind: [f32; 3]) {
        let delta_squared = delta * delta;
        let velocity_scale = 1.0 / (3.0 * delta);
        for triangle in 0..self.indices.len() / 3 {
            let offset = triangle * 3;
            let a = self.indices[offset] as usize;
            let b = self.indices[offset + 1] as usize;
            let c = self.indices[offset + 2] as usize;
            let ai = a * 3;
            let bi = b * 3;
            let ci = c * 3;
            let abx = self.positions[bi] - self.positions[ai];
            let aby = self.positions[bi + 1] - self.positions[ai + 1];
            let abz = self.positions[bi + 2] - self.positions[ai + 2];
            let acx = self.positions[ci] - self.positions[ai];
            let acy = self.positions[ci + 1] - self.positions[ai + 1];
            let acz = self.positions[ci + 2] - self.positions[ai + 2];
            let cross_x = aby * acz - abz * acy;
            let cross_y = abz * acx - abx * acz;
            let cross_z = abx * acy - aby * acx;
            let double_area = square_root(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
            if double_area < EPSILON {
                continue;
            }
            let nx = cross_x / double_area;
            let ny = cross_y / double_area;
            let nz = cross_z / double_area;
            let vx = (self.positions[ai] - self.previous[ai]
                + self.positions[bi] - self.previous[bi]
                + self.positions[ci] - self.previous[ci])
                * velocity_scale;
            let vy = (self.positions[ai + 1] - self.previous[ai + 1]
                + self.positions[bi + 1] - self.previous[bi + 1]
                + self.positions[ci + 1] - self.previous[ci + 1])
                * velocity_scale;
            let vz = (self.positions[ai + 2] - self.previous[ai + 2]
                + self.positions[bi + 2] - self.previous[bi + 2]
                + self.positions[ci + 2] - self.previous[ci + 2])
                * velocity_scale;
            let relative_x = wind[0] - vx;
            let relative_y = wind[1] - vy;
            let relative_z = wind[2] - vz;
            let normal_speed = relative_x * nx + relative_y * ny + relative_z * nz;
            let material = self.material_index(triangle);
            let drag = self.material_drags.get(material).copied().unwrap_or(self.material_drags[0]);
            let lift = self.material_lifts.get(material).copied().unwrap_or(self.material_lifts[0]);
            let drag_scale = normal_speed
                * normal_speed.abs()
                * double_area
                * 0.5
                * drag
                * delta_squared
                / 3.0;
            let lift_scale = normal_speed.abs() * double_area * 0.5 * lift * delta_squared / 3.0;
            for particle in [a, b, c] {
                if self.inverse_masses[particle] == 0.0 {
                    continue;
                }
                let particle_offset = particle * 3;
                let inverse_mass = self.inverse_masses[particle];
                self.positions[particle_offset] +=
                    (nx * drag_scale + relative_x * lift_scale) * inverse_mass;
                self.positions[particle_offset + 1] +=
                    (ny * drag_scale + relative_y * lift_scale) * inverse_mass;
                self.positions[particle_offset + 2] +=
                    (nz * drag_scale + relative_z * lift_scale) * inverse_mass;
            }
        }
    }

    fn limit_speed(&mut self, maximum_displacement: f32) {
        if !maximum_displacement.is_finite() {
            return;
        }
        let maximum_squared = maximum_displacement * maximum_displacement;
        for (particle, inverse_mass) in self.inverse_masses.iter().copied().enumerate() {
            if inverse_mass == 0.0 {
                continue;
            }
            let offset = particle * 3;
            let dx = self.positions[offset] - self.previous[offset];
            let dy = self.positions[offset + 1] - self.previous[offset + 1];
            let dz = self.positions[offset + 2] - self.previous[offset + 2];
            let distance_squared = dx * dx + dy * dy + dz * dz;
            if distance_squared <= maximum_squared {
                continue;
            }
            let scale = maximum_displacement / square_root(distance_squared);
            self.positions[offset] = self.previous[offset] + dx * scale;
            self.positions[offset + 1] = self.previous[offset + 1] + dy * scale;
            self.positions[offset + 2] = self.previous[offset + 2] + dz * scale;
        }
    }

    fn solve_tethers(&mut self) {
        for tether in 0..self.tether_particles.len() {
            let particle = self.tether_particles[tether] as usize;
            let anchor = self.tether_anchors[tether] as usize;
            let particle_offset = particle * 3;
            let anchor_offset = anchor * 3;
            let dx = self.positions[particle_offset] - self.positions[anchor_offset];
            let dy = self.positions[particle_offset + 1] - self.positions[anchor_offset + 1];
            let dz = self.positions[particle_offset + 2] - self.positions[anchor_offset + 2];
            let current_length = square_root(dx * dx + dy * dy + dz * dz);
            let maximum_length = self.tether_lengths[tether];
            if current_length <= maximum_length || current_length < EPSILON {
                continue;
            }
            let particle_weight = self.inverse_masses[particle];
            let anchor_weight = self.inverse_masses[anchor];
            let denominator = particle_weight + anchor_weight;
            if denominator < EPSILON {
                continue;
            }
            let correction = (current_length - maximum_length) / (current_length * denominator);
            self.positions[particle_offset] -= dx * correction * particle_weight;
            self.positions[particle_offset + 1] -= dy * correction * particle_weight;
            self.positions[particle_offset + 2] -= dz * correction * particle_weight;
            self.positions[anchor_offset] += dx * correction * anchor_weight;
            self.positions[anchor_offset + 1] += dy * correction * anchor_weight;
            self.positions[anchor_offset + 2] += dz * correction * anchor_weight;
        }
    }

    fn solve_distance_constraints(
        &mut self,
        delta: f32,
        stretch_compliance: f32,
        bend_compliance: f32,
    ) {
        let inverse_delta_squared = 1.0 / (delta * delta);
        for constraint in 0..self.edge_kinds.len() {
            let a = self.edge_particles[constraint * 2] as usize;
            let b = self.edge_particles[constraint * 2 + 1] as usize;
            let ai = a * 3;
            let bi = b * 3;
            let dx = self.positions[ai] - self.positions[bi];
            let dy = self.positions[ai + 1] - self.positions[bi + 1];
            let dz = self.positions[ai + 2] - self.positions[bi + 2];
            let length = square_root(dx * dx + dy * dy + dz * dz);
            if length < EPSILON {
                continue;
            }
            let wa = self.inverse_masses[a];
            let wb = self.inverse_masses[b];
            let compliance = if self.edge_kinds[constraint] == STRETCH_EDGE {
                stretch_compliance
            } else {
                bend_compliance
            };
            let alpha = compliance * inverse_delta_squared;
            let lambda = -(length - self.edge_rest_lengths[constraint]) / (wa + wb + alpha);
            let scale = lambda / length;
            self.positions[ai] += dx * scale * wa;
            self.positions[ai + 1] += dy * scale * wa;
            self.positions[ai + 2] += dz * scale * wa;
            self.positions[bi] -= dx * scale * wb;
            self.positions[bi + 1] -= dy * scale * wb;
            self.positions[bi + 2] -= dz * scale * wb;
        }
    }

    fn solve_area_constraints(&mut self, delta: f32, shear_compliance: f32) {
        let alpha = shear_compliance / (delta * delta);
        for triangle in 0..self.triangle_rest_areas.len() {
            let particle_offset = triangle * 3;
            let a = self.triangle_particles[particle_offset] as usize;
            let b = self.triangle_particles[particle_offset + 1] as usize;
            let c = self.triangle_particles[particle_offset + 2] as usize;
            let ai = a * 3;
            let bi = b * 3;
            let ci = c * 3;
            let ax = self.positions[ai];
            let ay = self.positions[ai + 1];
            let az = self.positions[ai + 2];
            let bx = self.positions[bi];
            let by = self.positions[bi + 1];
            let bz = self.positions[bi + 2];
            let cx = self.positions[ci];
            let cy = self.positions[ci + 1];
            let cz = self.positions[ci + 2];
            let cross_x = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
            let cross_y = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
            let cross_z = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            let double_area =
                square_root(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
            if double_area < EPSILON {
                continue;
            }
            let nx = cross_x / double_area;
            let ny = cross_y / double_area;
            let nz = cross_z / double_area;
            let gax = (by - cy) * nz - (bz - cz) * ny;
            let gay = (bz - cz) * nx - (bx - cx) * nz;
            let gaz = (bx - cx) * ny - (by - cy) * nx;
            let gbx = (cy - ay) * nz - (cz - az) * ny;
            let gby = (cz - az) * nx - (cx - ax) * nz;
            let gbz = (cx - ax) * ny - (cy - ay) * nx;
            let gcx = (ay - by) * nz - (az - bz) * ny;
            let gcy = (az - bz) * nx - (ax - bx) * nz;
            let gcz = (ax - bx) * ny - (ay - by) * nx;
            let wa = self.inverse_masses[a];
            let wb = self.inverse_masses[b];
            let wc = self.inverse_masses[c];
            let denominator = alpha
                + 0.25
                    * (wa * (gax * gax + gay * gay + gaz * gaz)
                        + wb * (gbx * gbx + gby * gby + gbz * gbz)
                        + wc * (gcx * gcx + gcy * gcy + gcz * gcz));
            if denominator < EPSILON {
                continue;
            }
            let lambda = -(double_area * 0.5 - self.triangle_rest_areas[triangle]) / denominator;
            let scale_a = lambda * wa * 0.5;
            let scale_b = lambda * wb * 0.5;
            let scale_c = lambda * wc * 0.5;
            self.positions[ai] += gax * scale_a;
            self.positions[ai + 1] += gay * scale_a;
            self.positions[ai + 2] += gaz * scale_a;
            self.positions[bi] += gbx * scale_b;
            self.positions[bi + 1] += gby * scale_b;
            self.positions[bi + 2] += gbz * scale_b;
            self.positions[ci] += gcx * scale_c;
            self.positions[ci + 1] += gcy * scale_c;
            self.positions[ci + 2] += gcz * scale_c;
        }
    }

    pub(crate) fn state(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.positions.len() * 2);
        result.extend_from_slice(&self.positions);
        result.extend_from_slice(&self.previous);
        result
    }
}

#[wasm_bindgen]
pub struct WasmSolver {
    pub(crate) bodies: Vec<Option<Body>>,
}

#[wasm_bindgen]
impl WasmSolver {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { bodies: Vec::new() }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_body(
        &mut self,
        id: u32,
        initial: &[f32],
        inverse_masses: &[f32],
        indices: &[u32],
        adjacency: &[u32],
        adjacency_offsets: &[u32],
        edge_kinds: &[u8],
        edge_particles: &[u32],
        edge_rest_lengths: &[f32],
        triangle_particles: &[u32],
        triangle_rest_areas: &[f32],
        tether_anchors: &[u32],
        tether_lengths: &[f32],
        tether_particles: &[u32],
        material_thicknesses: &[f32],
        material_frictions: &[f32],
        material_drags: &[f32],
        material_lifts: &[f32],
        triangle_material_indices: &[u16],
        collision_cell_size: f32,
        collision_layer: i32,
        collision_layer_axis: &[f32],
        filter_group: u32,
        filter_mask: u32,
        self_collision: bool,
    ) {
        assert_eq!(initial.len(), inverse_masses.len() * 3);
        assert_eq!(indices.len() % 3, 0);
        assert_eq!(adjacency_offsets.len(), inverse_masses.len() + 1);
        assert_eq!(edge_particles.len(), edge_kinds.len() * 2);
        assert_eq!(edge_rest_lengths.len(), edge_kinds.len());
        assert_eq!(triangle_particles.len(), triangle_rest_areas.len() * 3);
        assert_eq!(tether_anchors.len(), tether_particles.len());
        assert_eq!(tether_lengths.len(), tether_particles.len());
        assert!(!material_thicknesses.is_empty());
        assert!(!material_frictions.is_empty());
        assert!(!material_drags.is_empty());
        assert!(!material_lifts.is_empty());
        assert!(collision_layer_axis.is_empty() || collision_layer_axis.len() == 3);

        let index = id as usize;
        if self.bodies.len() <= index {
            self.bodies.resize_with(index + 1, || None);
        }
        let mut axis = [0.0; 3];
        if collision_layer_axis.len() == 3 {
            axis.copy_from_slice(collision_layer_axis);
        }
        self.bodies[index] = Some(Body {
            adjacency: adjacency.to_vec(),
            adjacency_offsets: adjacency_offsets.to_vec(),
            collision_cell_size,
            collision_layer,
            collision_layer_axis: axis,
            edge_kinds: edge_kinds.to_vec(),
            edge_particles: edge_particles.to_vec(),
            edge_rest_lengths: edge_rest_lengths.to_vec(),
            filter_group,
            filter_mask,
            has_collision_layer_axis: collision_layer_axis.len() == 3,
            indices: indices.to_vec(),
            initial: initial.to_vec(),
            inverse_masses: inverse_masses.to_vec(),
            material_drags: material_drags.to_vec(),
            material_frictions: material_frictions.to_vec(),
            material_lifts: material_lifts.to_vec(),
            material_thicknesses: material_thicknesses.to_vec(),
            maximum_self_collision_depenetration: f32::INFINITY,
            positions: initial.to_vec(),
            previous: initial.to_vec(),
            self_collision,
            tether_anchors: tether_anchors.to_vec(),
            tether_lengths: tether_lengths.to_vec(),
            tether_particles: tether_particles.to_vec(),
            triangle_material_indices: triangle_material_indices.to_vec(),
            triangle_particles: triangle_particles.to_vec(),
            triangle_rest_areas: triangle_rest_areas.to_vec(),
        });
    }

    pub fn has_body(&self, id: u32) -> bool {
        self.bodies
            .get(id as usize)
            .and_then(Option::as_ref)
            .is_some()
    }

    pub fn remove_body(&mut self, id: u32) {
        if let Some(body) = self.bodies.get_mut(id as usize) {
            *body = None;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve_structural(
        &mut self,
        id: u32,
        positions: &[f32],
        previous: &[f32],
        accelerations: &[f32],
        delta: f32,
        damping: f32,
        maximum_displacement: f32,
        maximum_depenetration: f32,
        stretch_compliance: f32,
        bend_compliance: f32,
        shear_compliance: f32,
        wind_x: f32,
        wind_y: f32,
        wind_z: f32,
        has_wind: bool,
    ) -> Vec<f32> {
        let body = self
            .bodies
            .get_mut(id as usize)
            .and_then(Option::as_mut)
            .expect("unknown cloth body");
        assert_eq!(positions.len(), body.positions.len());
        assert_eq!(previous.len(), body.previous.len());
        assert_eq!(accelerations.len(), body.positions.len());
        body.positions.copy_from_slice(positions);
        body.previous.copy_from_slice(previous);
        body.maximum_self_collision_depenetration = maximum_depenetration;
        body.integrate(accelerations, delta, damping);
        if has_wind {
            body.apply_aerodynamics(delta, [wind_x, wind_y, wind_z]);
        }
        body.limit_speed(maximum_displacement);
        body.solve_tethers();
        body.solve_distance_constraints(delta, stretch_compliance, bend_compliance);
        body.solve_area_constraints(delta, shear_compliance);
        body.state()
    }
}
