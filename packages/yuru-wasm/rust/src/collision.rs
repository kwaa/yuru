use alloc::vec::Vec;
use wasm_bindgen::prelude::*;

use crate::solver::{Body, WasmSolver};

const EPSILON: f32 = 1.0e-8;

#[derive(Clone, Copy)]
struct Candidate {
    body: usize,
    triangle_offset: usize,
}

#[derive(Clone, Copy)]
struct CellEntry {
    candidate: usize,
    key: u32,
}

#[derive(Clone, Copy)]
struct EdgeEntry {
    a: usize,
    b: usize,
    body: usize,
}

#[inline]
fn square_root(value: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        use core::arch::wasm32::{f32x4_extract_lane, f32x4_splat, f32x4_sqrt};
        return f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(value)));
    }
    #[cfg(not(target_arch = "wasm32"))]
    value.sqrt()
}

#[inline]
fn length(x: f32, y: f32, z: f32) -> f32 {
    square_root(x * x + y * y + z * z)
}

#[inline]
fn floor_to_i32(value: f32) -> i32 {
    let truncated = value as i32;
    if value < truncated as f32 {
        truncated.saturating_sub(1)
    } else {
        truncated
    }
}

#[inline]
fn ceil_to_i32(value: f32) -> i32 {
    let truncated = value as i32;
    if value > truncated as f32 {
        truncated.saturating_add(1)
    } else {
        truncated
    }
}

#[inline]
fn cell_key(x: f32, y: f32, z: f32, inverse_cell_size: f32) -> u32 {
    let cell_x = floor_to_i32(x * inverse_cell_size);
    let cell_y = floor_to_i32(y * inverse_cell_size);
    let cell_z = floor_to_i32(z * inverse_cell_size);
    (cell_x.wrapping_mul(73_856_093)
        ^ cell_y.wrapping_mul(19_349_663)
        ^ cell_z.wrapping_mul(83_492_791)) as u32
}

#[inline]
fn filters_collide(first: &Body, second: &Body) -> bool {
    (first.filter_group & second.filter_mask) != 0 && (second.filter_group & first.filter_mask) != 0
}

fn lower_bound(entries: &[CellEntry], key: u32) -> usize {
    let mut low = 0;
    let mut high = entries.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if entries[middle].key < key {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

fn upper_bound(entries: &[CellEntry], key: u32) -> usize {
    let mut low = 0;
    let mut high = entries.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if entries[middle].key <= key {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

#[inline]
fn is_adjacent(body: &Body, particle: usize, candidate: usize) -> bool {
    let start = body.adjacency_offsets[particle] as usize;
    let end = body.adjacency_offsets[particle + 1] as usize;
    body.adjacency[start..end]
        .binary_search(&(candidate as u32))
        .is_ok()
}

fn is_within_two_rings(body: &Body, particle: usize, candidate: usize) -> bool {
    if particle == candidate || is_adjacent(body, particle, candidate) {
        return true;
    }
    let start = body.adjacency_offsets[particle] as usize;
    let end = body.adjacency_offsets[particle + 1] as usize;
    for neighbor in &body.adjacency[start..end] {
        if is_adjacent(body, *neighbor as usize, candidate) {
            return true;
        }
    }
    false
}

/** Returns closest point xyz followed by barycentric weights. */
#[allow(clippy::too_many_arguments)]
fn closest_point_on_triangle(
    px: f32,
    py: f32,
    pz: f32,
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
) -> [f32; 6] {
    let abx = bx - ax;
    let aby = by - ay;
    let abz = bz - az;
    let acx = cx - ax;
    let acy = cy - ay;
    let acz = cz - az;
    let apx = px - ax;
    let apy = py - ay;
    let apz = pz - az;
    let d1 = abx * apx + aby * apy + abz * apz;
    let d2 = acx * apx + acy * apy + acz * apz;
    if d1 <= 0.0 && d2 <= 0.0 {
        return [ax, ay, az, 1.0, 0.0, 0.0];
    }

    let bpx = px - bx;
    let bpy = py - by;
    let bpz = pz - bz;
    let d3 = abx * bpx + aby * bpy + abz * bpz;
    let d4 = acx * bpx + acy * bpy + acz * bpz;
    if d3 >= 0.0 && d4 <= d3 {
        return [bx, by, bz, 0.0, 1.0, 0.0];
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [ax + v * abx, ay + v * aby, az + v * abz, 1.0 - v, v, 0.0];
    }

    let cpx = px - cx;
    let cpy = py - cy;
    let cpz = pz - cz;
    let d5 = abx * cpx + aby * cpy + abz * cpz;
    let d6 = acx * cpx + acy * cpy + acz * cpz;
    if d6 >= 0.0 && d5 <= d6 {
        return [cx, cy, cz, 0.0, 0.0, 1.0];
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [ax + w * acx, ay + w * acy, az + w * acz, 1.0 - w, 0.0, w];
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [
            bx + w * (cx - bx),
            by + w * (cy - by),
            bz + w * (cz - bz),
            0.0,
            1.0 - w,
            w,
        ];
    }

    let denominator = 1.0 / (va + vb + vc);
    let v = vb * denominator;
    let w = vc * denominator;
    let u = 1.0 - v - w;
    [
        ax * u + bx * v + cx * w,
        ay * u + by * v + cy * w,
        az * u + bz * v + cz * w,
        u,
        v,
        w,
    ]
}

#[inline]
fn limited_depenetration(
    penetration: f32,
    initial_penetration: f32,
    maximum_correction: f32,
) -> f32 {
    if !maximum_correction.is_finite() || initial_penetration <= 0.0 {
        return penetration;
    }
    let predicted_penetration = (penetration - initial_penetration).max(0.0);
    penetration.min(predicted_penetration + initial_penetration.min(maximum_correction))
}

fn closest_segment_segment(p1: [f32; 3], q1: [f32; 3], p2: [f32; 3], q2: [f32; 3]) -> [f32; 8] {
    let d1x = q1[0] - p1[0];
    let d1y = q1[1] - p1[1];
    let d1z = q1[2] - p1[2];
    let d2x = q2[0] - p2[0];
    let d2y = q2[1] - p2[1];
    let d2z = q2[2] - p2[2];
    let rx = p1[0] - p2[0];
    let ry = p1[1] - p2[1];
    let rz = p1[2] - p2[2];
    let a = d1x * d1x + d1y * d1y + d1z * d1z;
    let e = d2x * d2x + d2y * d2y + d2z * d2z;
    let f = d2x * rx + d2y * ry + d2z * rz;
    let mut s = 0.0;
    let mut t = 0.0;
    if a <= EPSILON && e <= EPSILON {
        // Both segments degenerate to points.
    } else if a <= EPSILON {
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1x * rx + d1y * ry + d1z * rz;
        if e <= EPSILON {
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1x * d2x + d1y * d2y + d1z * d2z;
            let denominator = a * e - b * b;
            if denominator != 0.0 {
                s = ((b * f - c * e) / denominator).clamp(0.0, 1.0);
            }
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    [
        p1[0] + d1x * s,
        p1[1] + d1y * s,
        p1[2] + d1z * s,
        p2[0] + d2x * t,
        p2[1] + d2y * t,
        p2[2] + d2z * t,
        s,
        t,
    ]
}

impl WasmSolver {
    fn solve_vertex_triangle(
        &mut self,
        vertex_body_index: usize,
        particle: usize,
        triangle_body_index: usize,
        triangle_offset: usize,
    ) {
        let p = particle * 3;
        let a =
            self.bodies[triangle_body_index].as_ref().unwrap().indices[triangle_offset] as usize;
        let b = self.bodies[triangle_body_index].as_ref().unwrap().indices[triangle_offset + 1]
            as usize;
        let c = self.bodies[triangle_body_index].as_ref().unwrap().indices[triangle_offset + 2]
            as usize;
        let ai = a * 3;
        let bi = b * 3;
        let ci = c * 3;

        let vertex = self.bodies[vertex_body_index].as_ref().unwrap();
        let triangle = self.bodies[triangle_body_index].as_ref().unwrap();
        let closest = closest_point_on_triangle(
            vertex.positions[p],
            vertex.positions[p + 1],
            vertex.positions[p + 2],
            triangle.positions[ai],
            triangle.positions[ai + 1],
            triangle.positions[ai + 2],
            triangle.positions[bi],
            triangle.positions[bi + 1],
            triangle.positions[bi + 2],
            triangle.positions[ci],
            triangle.positions[ci + 1],
            triangle.positions[ci + 2],
        );
        let mut dx = vertex.positions[p] - closest[0];
        let mut dy = vertex.positions[p + 1] - closest[1];
        let mut dz = vertex.positions[p + 2] - closest[2];
        let mut separation = length(dx, dy, dz);
        let thickness =
            vertex.material_thicknesses[0] + triangle.thickness_for_triangle(triangle_offset / 3);
        let ordered_layers = vertex_body_index != triangle_body_index
            && vertex.collision_layer != triangle.collision_layer;
        let mut contact_distance = thickness;
        let mut penetration = thickness - separation;
        let previous_closest = closest_point_on_triangle(
            vertex.previous[p],
            vertex.previous[p + 1],
            vertex.previous[p + 2],
            triangle.previous[ai],
            triangle.previous[ai + 1],
            triangle.previous[ai + 2],
            triangle.previous[bi],
            triangle.previous[bi + 1],
            triangle.previous[bi + 2],
            triangle.previous[ci],
            triangle.previous[ci + 1],
            triangle.previous[ci + 2],
        );
        let previous_distance;
        if ordered_layers {
            if separation >= thickness * 2.0 {
                return;
            }
            let abx = triangle.positions[bi] - triangle.positions[ai];
            let aby = triangle.positions[bi + 1] - triangle.positions[ai + 1];
            let abz = triangle.positions[bi + 2] - triangle.positions[ai + 2];
            let acx = triangle.positions[ci] - triangle.positions[ai];
            let acy = triangle.positions[ci + 1] - triangle.positions[ai + 1];
            let acz = triangle.positions[ci + 2] - triangle.positions[ai + 2];
            let side = if vertex.collision_layer > triangle.collision_layer {
                1.0
            } else {
                -1.0
            };
            dx = (aby * acz - abz * acy) * side;
            dy = (abz * acx - abx * acz) * side;
            dz = (abx * acy - aby * acx) * side;
            let layer_axis = if triangle.has_collision_layer_axis {
                Some(triangle.collision_layer_axis)
            } else if vertex.has_collision_layer_axis {
                Some(vertex.collision_layer_axis)
            } else {
                None
            };
            if let Some(axis) = layer_axis {
                let along_axis = dx * axis[0] + dy * axis[1] + dz * axis[2];
                dx -= axis[0] * along_axis;
                dy -= axis[1] * along_axis;
                dz -= axis[2] * along_axis;
            }
            let normal_length = length(dx, dy, dz);
            if normal_length < EPSILON {
                return;
            }
            dx /= normal_length;
            dy /= normal_length;
            dz /= normal_length;
            let rest_closest = closest_point_on_triangle(
                vertex.initial[p],
                vertex.initial[p + 1],
                vertex.initial[p + 2],
                triangle.initial[ai],
                triangle.initial[ai + 1],
                triangle.initial[ai + 2],
                triangle.initial[bi],
                triangle.initial[bi + 1],
                triangle.initial[bi + 2],
                triangle.initial[ci],
                triangle.initial[ci + 1],
                triangle.initial[ci + 2],
            );
            let rest_distance = (vertex.initial[p] - rest_closest[0]) * dx
                + (vertex.initial[p + 1] - rest_closest[1]) * dy
                + (vertex.initial[p + 2] - rest_closest[2]) * dz;
            contact_distance = thickness.min(rest_distance).max(0.0);
            let oriented_distance = (vertex.positions[p] - closest[0]) * dx
                + (vertex.positions[p + 1] - closest[1]) * dy
                + (vertex.positions[p + 2] - closest[2]) * dz;
            if oriented_distance >= contact_distance {
                return;
            }
            penetration = contact_distance - oriented_distance;
            previous_distance = (vertex.previous[p] - previous_closest[0]) * dx
                + (vertex.previous[p + 1] - previous_closest[1]) * dy
                + (vertex.previous[p + 2] - previous_closest[2]) * dz;
            separation = 1.0;
        } else {
            if separation >= thickness {
                return;
            }
            previous_distance = length(
                vertex.previous[p] - previous_closest[0],
                vertex.previous[p + 1] - previous_closest[1],
                vertex.previous[p + 2] - previous_closest[2],
            );
        }
        let initial_penetration = (contact_distance - previous_distance).max(0.0);
        if vertex_body_index == triangle_body_index {
            let rest_closest = closest_point_on_triangle(
                vertex.initial[p],
                vertex.initial[p + 1],
                vertex.initial[p + 2],
                triangle.initial[ai],
                triangle.initial[ai + 1],
                triangle.initial[ai + 2],
                triangle.initial[bi],
                triangle.initial[bi + 1],
                triangle.initial[bi + 2],
                triangle.initial[ci],
                triangle.initial[ci + 1],
                triangle.initial[ci + 2],
            );
            let rest_distance = length(
                vertex.initial[p] - rest_closest[0],
                vertex.initial[p + 1] - rest_closest[1],
                vertex.initial[p + 2] - rest_closest[2],
            );
            if rest_distance < thickness * 1.25 {
                return;
            }
        }
        if separation < EPSILON {
            let abx = triangle.positions[bi] - triangle.positions[ai];
            let aby = triangle.positions[bi + 1] - triangle.positions[ai + 1];
            let abz = triangle.positions[bi + 2] - triangle.positions[ai + 2];
            let acx = triangle.positions[ci] - triangle.positions[ai];
            let acy = triangle.positions[ci + 1] - triangle.positions[ai + 1];
            let acz = triangle.positions[ci + 2] - triangle.positions[ai + 2];
            dx = aby * acz - abz * acy;
            dy = abz * acx - abx * acz;
            dz = abx * acy - aby * acx;
            let normal_length = length(dx, dy, dz);
            if normal_length < EPSILON {
                dx = 0.0;
                dy = 1.0;
                dz = 0.0;
            } else {
                dx /= normal_length;
                dy /= normal_length;
                dz /= normal_length;
            }
            separation = 1.0;
        }
        let nx = dx / separation;
        let ny = dy / separation;
        let nz = dz / separation;
        let wp = vertex.inverse_masses[particle];
        let wa = triangle.inverse_masses[a];
        let wb = triangle.inverse_masses[b];
        let wc = triangle.inverse_masses[c];
        let denominator = wp
            + wa * closest[3] * closest[3]
            + wb * closest[4] * closest[4]
            + wc * closest[5] * closest[5];
        if denominator < EPSILON {
            return;
        }
        let maximum_depenetration = if vertex_body_index == triangle_body_index {
            vertex.maximum_self_collision_depenetration
        } else {
            vertex
                .maximum_self_collision_depenetration
                .min(triangle.maximum_self_collision_depenetration)
        };
        let normal_correction =
            limited_depenetration(penetration, initial_penetration, maximum_depenetration);
        let correction = normal_correction / denominator;
        let particle_correction = [
            nx * correction * wp,
            ny * correction * wp,
            nz * correction * wp,
        ];
        let barycentrics = [closest[3], closest[4], closest[5]];
        let triangle_particles = [a, b, c];
        let triangle_weights = [wa, wb, wc];
        let vertex_friction = vertex.material_frictions[0];
        let triangle_friction = triangle.friction_for_triangle(triangle_offset / 3);

        {
            let vertex = self.bodies[vertex_body_index].as_mut().unwrap();
            vertex.positions[p] += particle_correction[0];
            vertex.positions[p + 1] += particle_correction[1];
            vertex.positions[p + 2] += particle_correction[2];
            vertex.previous[p] += particle_correction[0];
            vertex.previous[p + 1] += particle_correction[1];
            vertex.previous[p + 2] += particle_correction[2];
        }
        for item in 0..3 {
            let offset = triangle_particles[item] * 3;
            let scale = correction * barycentrics[item] * triangle_weights[item];
            let correction_x = nx * scale;
            let correction_y = ny * scale;
            let correction_z = nz * scale;
            let triangle = self.bodies[triangle_body_index].as_mut().unwrap();
            triangle.positions[offset] -= correction_x;
            triangle.positions[offset + 1] -= correction_y;
            triangle.positions[offset + 2] -= correction_z;
            triangle.previous[offset] -= correction_x;
            triangle.previous[offset + 1] -= correction_y;
            triangle.previous[offset + 2] -= correction_z;
        }

        let vertex = self.bodies[vertex_body_index].as_ref().unwrap();
        let triangle = self.bodies[triangle_body_index].as_ref().unwrap();
        let relative_x = vertex.positions[p]
            - vertex.previous[p]
            - (triangle.positions[ai] - triangle.previous[ai]) * closest[3]
            - (triangle.positions[bi] - triangle.previous[bi]) * closest[4]
            - (triangle.positions[ci] - triangle.previous[ci]) * closest[5];
        let relative_y = vertex.positions[p + 1]
            - vertex.previous[p + 1]
            - (triangle.positions[ai + 1] - triangle.previous[ai + 1]) * closest[3]
            - (triangle.positions[bi + 1] - triangle.previous[bi + 1]) * closest[4]
            - (triangle.positions[ci + 1] - triangle.previous[ci + 1]) * closest[5];
        let relative_z = vertex.positions[p + 2]
            - vertex.previous[p + 2]
            - (triangle.positions[ai + 2] - triangle.previous[ai + 2]) * closest[3]
            - (triangle.positions[bi + 2] - triangle.previous[bi + 2]) * closest[4]
            - (triangle.positions[ci + 2] - triangle.previous[ci + 2]) * closest[5];
        let normal_velocity = relative_x * nx + relative_y * ny + relative_z * nz;
        let tangent_x = relative_x - nx * normal_velocity;
        let tangent_y = relative_y - ny * normal_velocity;
        let tangent_z = relative_z - nz * normal_velocity;
        let tangent_length = length(tangent_x, tangent_y, tangent_z);
        let friction = ((vertex_friction + triangle_friction) * 0.5).max(0.0);
        let tangent_scale = if tangent_length < EPSILON {
            0.0
        } else {
            (friction * normal_correction / tangent_length).min(1.0)
        };
        let velocity_correction_x =
            (nx * (-normal_velocity).max(0.0) - tangent_x * tangent_scale) / denominator;
        let velocity_correction_y =
            (ny * (-normal_velocity).max(0.0) - tangent_y * tangent_scale) / denominator;
        let velocity_correction_z =
            (nz * (-normal_velocity).max(0.0) - tangent_z * tangent_scale) / denominator;
        {
            let vertex = self.bodies[vertex_body_index].as_mut().unwrap();
            vertex.previous[p] -= velocity_correction_x * wp;
            vertex.previous[p + 1] -= velocity_correction_y * wp;
            vertex.previous[p + 2] -= velocity_correction_z * wp;
        }
        for item in 0..3 {
            let offset = triangle_particles[item] * 3;
            let scale = barycentrics[item] * triangle_weights[item];
            let triangle = self.bodies[triangle_body_index].as_mut().unwrap();
            triangle.previous[offset] += velocity_correction_x * scale;
            triangle.previous[offset + 1] += velocity_correction_y * scale;
            triangle.previous[offset + 2] += velocity_correction_z * scale;
        }
    }

    fn solve_edge_edge(&mut self, first: EdgeEntry, second: EdgeEntry) {
        let ai = first.a * 3;
        let bi = first.b * 3;
        let ci = second.a * 3;
        let di = second.b * 3;
        let first_body = self.bodies[first.body].as_ref().unwrap();
        let second_body = self.bodies[second.body].as_ref().unwrap();
        let closest = closest_segment_segment(
            [
                first_body.positions[ai],
                first_body.positions[ai + 1],
                first_body.positions[ai + 2],
            ],
            [
                first_body.positions[bi],
                first_body.positions[bi + 1],
                first_body.positions[bi + 2],
            ],
            [
                second_body.positions[ci],
                second_body.positions[ci + 1],
                second_body.positions[ci + 2],
            ],
            [
                second_body.positions[di],
                second_body.positions[di + 1],
                second_body.positions[di + 2],
            ],
        );
        let mut dx = closest[0] - closest[3];
        let mut dy = closest[1] - closest[4];
        let mut dz = closest[2] - closest[5];
        let separation = length(dx, dy, dz);
        let target = first_body.material_thicknesses[0] + second_body.material_thicknesses[0];
        if separation >= target {
            return;
        }
        let previous_closest = closest_segment_segment(
            [
                first_body.previous[ai],
                first_body.previous[ai + 1],
                first_body.previous[ai + 2],
            ],
            [
                first_body.previous[bi],
                first_body.previous[bi + 1],
                first_body.previous[bi + 2],
            ],
            [
                second_body.previous[ci],
                second_body.previous[ci + 1],
                second_body.previous[ci + 2],
            ],
            [
                second_body.previous[di],
                second_body.previous[di + 1],
                second_body.previous[di + 2],
            ],
        );
        let previous_dx = previous_closest[0] - previous_closest[3];
        let previous_dy = previous_closest[1] - previous_closest[4];
        let previous_dz = previous_closest[2] - previous_closest[5];
        let previous_length = length(previous_dx, previous_dy, previous_dz);
        let same_body = first.body == second.body;
        let mut rest_dx = 0.0;
        let mut rest_dy = 0.0;
        let mut rest_dz = 0.0;
        let initial_penetration;
        if same_body {
            let rest_closest = closest_segment_segment(
                [
                    first_body.initial[ai],
                    first_body.initial[ai + 1],
                    first_body.initial[ai + 2],
                ],
                [
                    first_body.initial[bi],
                    first_body.initial[bi + 1],
                    first_body.initial[bi + 2],
                ],
                [
                    second_body.initial[ci],
                    second_body.initial[ci + 1],
                    second_body.initial[ci + 2],
                ],
                [
                    second_body.initial[di],
                    second_body.initial[di + 1],
                    second_body.initial[di + 2],
                ],
            );
            rest_dx = rest_closest[0] - rest_closest[3];
            rest_dy = rest_closest[1] - rest_closest[4];
            rest_dz = rest_closest[2] - rest_closest[5];
            if length(rest_dx, rest_dy, rest_dz) < target * 1.25 {
                return;
            }
            initial_penetration = (target - previous_length).max(0.0);
        } else {
            initial_penetration = (target - previous_length).max(0.0);
        }
        if separation >= EPSILON {
            dx /= separation;
            dy /= separation;
            dz /= separation;
            if previous_length >= EPSILON
                && dx * previous_dx + dy * previous_dy + dz * previous_dz < 0.0
            {
                dx = -dx;
                dy = -dy;
                dz = -dz;
            }
        } else if previous_length >= EPSILON {
            dx = previous_dx / previous_length;
            dy = previous_dy / previous_length;
            dz = previous_dz / previous_length;
        } else {
            let rest_length = length(rest_dx, rest_dy, rest_dz);
            if rest_length >= EPSILON {
                dx = rest_dx / rest_length;
                dy = rest_dy / rest_length;
                dz = rest_dz / rest_length;
            } else {
                let first_x = first_body.positions[bi] - first_body.positions[ai];
                let first_y = first_body.positions[bi + 1] - first_body.positions[ai + 1];
                let first_z = first_body.positions[bi + 2] - first_body.positions[ai + 2];
                let second_x = second_body.positions[di] - second_body.positions[ci];
                let second_y = second_body.positions[di + 1] - second_body.positions[ci + 1];
                let second_z = second_body.positions[di + 2] - second_body.positions[ci + 2];
                dx = first_y * second_z - first_z * second_y;
                dy = first_z * second_x - first_x * second_z;
                dz = first_x * second_y - first_y * second_x;
                let normal_length = length(dx, dy, dz);
                if normal_length < EPSILON {
                    dx = 0.0;
                    dy = 1.0;
                    dz = 0.0;
                } else {
                    dx /= normal_length;
                    dy /= normal_length;
                    dz /= normal_length;
                }
            }
        }
        let interpolations = [1.0 - closest[6], closest[6], 1.0 - closest[7], closest[7]];
        let particle_indices = [first.a, first.b, second.a, second.b];
        let body_indices = [first.body, first.body, second.body, second.body];
        let directions = [1.0, 1.0, -1.0, -1.0];
        let weights = [
            first_body.inverse_masses[first.a] * interpolations[0] * interpolations[0],
            first_body.inverse_masses[first.b] * interpolations[1] * interpolations[1],
            second_body.inverse_masses[second.a] * interpolations[2] * interpolations[2],
            second_body.inverse_masses[second.b] * interpolations[3] * interpolations[3],
        ];
        let denominator = weights[0] + weights[1] + weights[2] + weights[3];
        if denominator < EPSILON {
            return;
        }
        let maximum_depenetration = if same_body {
            first_body.maximum_self_collision_depenetration
        } else {
            first_body
                .maximum_self_collision_depenetration
                .min(second_body.maximum_self_collision_depenetration)
        };
        let normal_correction = limited_depenetration(
            target - separation,
            initial_penetration,
            maximum_depenetration,
        );
        let correction = normal_correction / denominator;
        let first_friction = first_body.material_frictions[0];
        let second_friction = second_body.material_frictions[0];

        for item in 0..4 {
            let body = self.bodies[body_indices[item]].as_mut().unwrap();
            let particle = particle_indices[item];
            let offset = particle * 3;
            let scale = correction
                * body.inverse_masses[particle]
                * interpolations[item]
                * directions[item];
            let correction_x = dx * scale;
            let correction_y = dy * scale;
            let correction_z = dz * scale;
            body.positions[offset] += correction_x;
            body.positions[offset + 1] += correction_y;
            body.positions[offset + 2] += correction_z;
            body.previous[offset] += correction_x;
            body.previous[offset + 1] += correction_y;
            body.previous[offset + 2] += correction_z;
        }

        let first_body = self.bodies[first.body].as_ref().unwrap();
        let second_body = self.bodies[second.body].as_ref().unwrap();
        let velocity_a = [
            first_body.positions[ai] - first_body.previous[ai],
            first_body.positions[ai + 1] - first_body.previous[ai + 1],
            first_body.positions[ai + 2] - first_body.previous[ai + 2],
        ];
        let velocity_b = [
            first_body.positions[bi] - first_body.previous[bi],
            first_body.positions[bi + 1] - first_body.previous[bi + 1],
            first_body.positions[bi + 2] - first_body.previous[bi + 2],
        ];
        let velocity_c = [
            second_body.positions[ci] - second_body.previous[ci],
            second_body.positions[ci + 1] - second_body.previous[ci + 1],
            second_body.positions[ci + 2] - second_body.previous[ci + 2],
        ];
        let velocity_d = [
            second_body.positions[di] - second_body.previous[di],
            second_body.positions[di + 1] - second_body.previous[di + 1],
            second_body.positions[di + 2] - second_body.previous[di + 2],
        ];
        let relative_x = velocity_a[0] * interpolations[0] + velocity_b[0] * interpolations[1]
            - velocity_c[0] * interpolations[2]
            - velocity_d[0] * interpolations[3];
        let relative_y = velocity_a[1] * interpolations[0] + velocity_b[1] * interpolations[1]
            - velocity_c[1] * interpolations[2]
            - velocity_d[1] * interpolations[3];
        let relative_z = velocity_a[2] * interpolations[0] + velocity_b[2] * interpolations[1]
            - velocity_c[2] * interpolations[2]
            - velocity_d[2] * interpolations[3];
        let normal_velocity = relative_x * dx + relative_y * dy + relative_z * dz;
        let tangent_x = relative_x - dx * normal_velocity;
        let tangent_y = relative_y - dy * normal_velocity;
        let tangent_z = relative_z - dz * normal_velocity;
        let tangent_length = length(tangent_x, tangent_y, tangent_z);
        let friction = ((first_friction + second_friction) * 0.5).max(0.0);
        let tangent_scale = if tangent_length < EPSILON {
            0.0
        } else {
            (friction * normal_correction / tangent_length).min(1.0)
        };
        let velocity_correction_x =
            (dx * (-normal_velocity).max(0.0) - tangent_x * tangent_scale) / denominator;
        let velocity_correction_y =
            (dy * (-normal_velocity).max(0.0) - tangent_y * tangent_scale) / denominator;
        let velocity_correction_z =
            (dz * (-normal_velocity).max(0.0) - tangent_z * tangent_scale) / denominator;
        for item in 0..4 {
            let body = self.bodies[body_indices[item]].as_mut().unwrap();
            let particle = particle_indices[item];
            let offset = particle * 3;
            let scale = body.inverse_masses[particle] * interpolations[item] * directions[item];
            body.previous[offset] -= velocity_correction_x * scale;
            body.previous[offset + 1] -= velocity_correction_y * scale;
            body.previous[offset + 2] -= velocity_correction_z * scale;
        }
    }
}

#[wasm_bindgen]
impl WasmSolver {
    pub fn sync_body(
        &mut self,
        id: u32,
        positions: &[f32],
        previous: &[f32],
        maximum_depenetration: f32,
    ) {
        let body = self
            .bodies
            .get_mut(id as usize)
            .and_then(Option::as_mut)
            .expect("unknown cloth body");
        assert_eq!(positions.len(), body.positions.len());
        assert_eq!(previous.len(), body.previous.len());
        body.positions.copy_from_slice(positions);
        body.previous.copy_from_slice(previous);
        body.maximum_self_collision_depenetration = maximum_depenetration;
    }

    pub fn body_state(&self, id: u32) -> Vec<f32> {
        self.bodies
            .get(id as usize)
            .and_then(Option::as_ref)
            .expect("unknown cloth body")
            .state()
    }

    pub fn solve_body_collisions(&mut self, max_candidates: u32) {
        let active: Vec<usize> = self
            .bodies
            .iter()
            .enumerate()
            .filter_map(|(index, body)| body.as_ref().map(|_| index))
            .collect();
        if active.is_empty() {
            return;
        }
        let mut maximum_thickness = 0.0_f32;
        let mut maximum_cell_size = 0.0_f32;
        for body_index in &active {
            let body = self.bodies[*body_index].as_ref().unwrap();
            maximum_cell_size = maximum_cell_size.max(body.collision_cell_size);
            for thickness in &body.material_thicknesses {
                maximum_thickness = maximum_thickness.max(*thickness);
            }
        }
        let inverse_cell_size = 1.0 / maximum_cell_size;
        let mut candidates = Vec::new();
        let mut cells = Vec::new();

        for body_index in &active {
            let body = self.bodies[*body_index].as_ref().unwrap();
            for triangle_offset in (0..body.indices.len()).step_by(3) {
                let candidate = candidates.len();
                candidates.push(Candidate {
                    body: *body_index,
                    triangle_offset,
                });
                let a = body.indices[triangle_offset] as usize * 3;
                let b = body.indices[triangle_offset + 1] as usize * 3;
                let c = body.indices[triangle_offset + 2] as usize * 3;
                let from_x = floor_to_i32(
                    (body.positions[a]
                        .min(body.positions[b])
                        .min(body.positions[c])
                        - maximum_thickness)
                        * inverse_cell_size,
                );
                let from_y = floor_to_i32(
                    (body.positions[a + 1]
                        .min(body.positions[b + 1])
                        .min(body.positions[c + 1])
                        - maximum_thickness)
                        * inverse_cell_size,
                );
                let from_z = floor_to_i32(
                    (body.positions[a + 2]
                        .min(body.positions[b + 2])
                        .min(body.positions[c + 2])
                        - maximum_thickness)
                        * inverse_cell_size,
                );
                let to_x = ceil_to_i32(
                    (body.positions[a]
                        .max(body.positions[b])
                        .max(body.positions[c])
                        + maximum_thickness)
                        * inverse_cell_size,
                );
                let to_y = ceil_to_i32(
                    (body.positions[a + 1]
                        .max(body.positions[b + 1])
                        .max(body.positions[c + 1])
                        + maximum_thickness)
                        * inverse_cell_size,
                );
                let to_z = ceil_to_i32(
                    (body.positions[a + 2]
                        .max(body.positions[b + 2])
                        .max(body.positions[c + 2])
                        + maximum_thickness)
                        * inverse_cell_size,
                );
                let mut inserted = 0;
                let mut x = from_x;
                while x <= to_x && inserted < 64 {
                    let mut y = from_y;
                    while y <= to_y && inserted < 64 {
                        let mut z = from_z;
                        while z <= to_z && inserted < 64 {
                            cells.push(CellEntry {
                                candidate,
                                key: (x.wrapping_mul(73_856_093)
                                    ^ y.wrapping_mul(19_349_663)
                                    ^ z.wrapping_mul(83_492_791))
                                    as u32,
                            });
                            inserted += 1;
                            z = z.saturating_add(1);
                        }
                        y = y.saturating_add(1);
                    }
                    x = x.saturating_add(1);
                }
            }
        }
        cells.sort_by_key(|entry| entry.key);

        for vertex_body_index in active {
            let particle_count = self.bodies[vertex_body_index]
                .as_ref()
                .unwrap()
                .inverse_masses
                .len();
            for particle in 0..particle_count {
                let vertex_body = self.bodies[vertex_body_index].as_ref().unwrap();
                if vertex_body.inverse_masses[particle] == 0.0 {
                    continue;
                }
                let particle_offset = particle * 3;
                let key = cell_key(
                    vertex_body.positions[particle_offset],
                    vertex_body.positions[particle_offset + 1],
                    vertex_body.positions[particle_offset + 2],
                    inverse_cell_size,
                );
                let start = lower_bound(&cells, key);
                let end = upper_bound(&cells, key);
                if start == end {
                    continue;
                }
                for same_body_pass in [false, true] {
                    let mut tested = 0_u32;
                    for entry in &cells[start..end] {
                        if tested >= max_candidates {
                            break;
                        }
                        let candidate = candidates[entry.candidate];
                        let same_body = candidate.body == vertex_body_index;
                        if same_body != same_body_pass {
                            continue;
                        }
                        let vertex_body = self.bodies[vertex_body_index].as_ref().unwrap();
                        let triangle_body = self.bodies[candidate.body].as_ref().unwrap();
                        if same_body_pass && !vertex_body.self_collision {
                            continue;
                        }
                        if !same_body_pass && !filters_collide(vertex_body, triangle_body) {
                            continue;
                        }
                        let a = triangle_body.indices[candidate.triangle_offset] as usize;
                        let b = triangle_body.indices[candidate.triangle_offset + 1] as usize;
                        let c = triangle_body.indices[candidate.triangle_offset + 2] as usize;
                        if same_body_pass
                            && (particle == a
                                || particle == b
                                || particle == c
                                || is_within_two_rings(vertex_body, particle, a)
                                || is_within_two_rings(vertex_body, particle, b)
                                || is_within_two_rings(vertex_body, particle, c))
                        {
                            continue;
                        }
                        tested += 1;
                        self.solve_vertex_triangle(
                            vertex_body_index,
                            particle,
                            candidate.body,
                            candidate.triangle_offset,
                        );
                    }
                }
            }
        }
    }

    pub fn solve_edge_collisions(&mut self, max_candidates: u32) {
        let active: Vec<usize> = self
            .bodies
            .iter()
            .enumerate()
            .filter_map(|(index, body)| body.as_ref().map(|_| index))
            .collect();
        if active.is_empty() {
            return;
        }
        let mut cell_size = 0.01_f32;
        for body_index in &active {
            cell_size = cell_size.max(
                self.bodies[*body_index]
                    .as_ref()
                    .unwrap()
                    .collision_cell_size,
            );
        }
        let inverse_cell_size = 1.0 / cell_size;
        let mut entries = Vec::new();
        let mut cells = Vec::new();
        for body_index in active {
            let body = self.bodies[body_index].as_ref().unwrap();
            for edge in 0..body.edge_kinds.len() {
                if body.edge_kinds[edge] != 0 {
                    continue;
                }
                let a = body.edge_particles[edge * 2] as usize;
                let b = body.edge_particles[edge * 2 + 1] as usize;
                let ai = a * 3;
                let bi = b * 3;
                let entry = entries.len();
                entries.push(EdgeEntry {
                    a,
                    b,
                    body: body_index,
                });
                cells.push(CellEntry {
                    candidate: entry,
                    key: cell_key(
                        (body.positions[ai] + body.positions[bi]) * 0.5,
                        (body.positions[ai + 1] + body.positions[bi + 1]) * 0.5,
                        (body.positions[ai + 2] + body.positions[bi + 2]) * 0.5,
                        inverse_cell_size,
                    ),
                });
            }
        }
        cells.sort_by_key(|entry| entry.key);
        let mut start = 0;
        while start < cells.len() {
            let key = cells[start].key;
            let end = upper_bound(&cells, key);
            let list = &cells[start..end];
            let maximum_tests = max_candidates.saturating_mul(list.len() as u32);
            let mut tested = 0_u32;
            for first_index in 0..list.len() {
                for second_index in first_index + 1..list.len() {
                    if tested >= maximum_tests {
                        break;
                    }
                    tested += 1;
                    let first = entries[list[first_index].candidate];
                    let second = entries[list[second_index].candidate];
                    let first_body = self.bodies[first.body].as_ref().unwrap();
                    let second_body = self.bodies[second.body].as_ref().unwrap();
                    if first.body == second.body {
                        if !first_body.self_collision
                            || is_within_two_rings(first_body, first.a, second.a)
                            || is_within_two_rings(first_body, first.a, second.b)
                            || is_within_two_rings(first_body, first.b, second.a)
                            || is_within_two_rings(first_body, first.b, second.b)
                        {
                            continue;
                        }
                    } else if !filters_collide(first_body, second_body) {
                        continue;
                    } else if first_body.collision_layer != second_body.collision_layer {
                        continue;
                    }
                    self.solve_edge_edge(first, second);
                }
            }
            start = end;
        }
    }
}
