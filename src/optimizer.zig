const std = @import("std");

const Matrix = @import("matrix.zig");
const Symbol = @import("symbol.zig");

pub fn MomentumSgdOptimizer(learning_rate: comptime_float, momentum_decay_coefficient: comptime_float) type {
    return struct {
        parameters: []Symbol,
        accumulated_gradients: []Matrix,

        const Self = @This();

        pub fn initialize(allocator: std.mem.Allocator, parameters: []Symbol) !Self {
            const ret = Self{
                .parameters = parameters,
                .accumulated_gradients = try allocator.alloc(Matrix, parameters.len),
            };

            for (ret.accumulated_gradients, ret.parameters) |*accumulated_gradient, parameter| {
                accumulated_gradient.* = try Matrix.initialize(allocator, parameter.gradient.row_count, parameter.gradient.column_count);
                @memset(accumulated_gradient.entries, 0);
            }

            return ret;
        }

        pub fn updateParameters(self: Self) void {
            for (self.parameters, self.accumulated_gradients) |parameter, accumulated_gradient| {
                for (parameter.value.entries, accumulated_gradient.entries, parameter.gradient.entries) |*value_entry, *accumulated_gradient_entry, gradient_entry| {
                    accumulated_gradient_entry.* = momentum_decay_coefficient * accumulated_gradient_entry.* + (1 - momentum_decay_coefficient) * gradient_entry;
                    value_entry.* -= learning_rate * accumulated_gradient_entry.*;
                }
            }
        }
    };
}

// https://arxiv.org/abs/2405.15682
pub fn ScheduleFreeSgdOptimizer(learning_rate: comptime_float, momentum_decay_coefficient: comptime_float) type {
    return struct {
        parameters: []Symbol, // y (stored as the model's internal parameters)
        x: []Matrix,
        z: []Matrix,

        t: f32,

        const Self = @This();

        pub fn initialize(allocator: std.mem.Allocator, parameters: []Symbol) !Self {
            const ret = Self{
                .parameters = parameters,
                .x = try allocator.alloc(Matrix, parameters.len),
                .z = try allocator.alloc(Matrix, parameters.len),

                .t = 0,
            };

            for (ret.x, ret.z, ret.parameters) |*x, *z, parameter| {
                x.* = try Matrix.initialize(allocator, parameter.value.row_count, parameter.value.column_count);
                z.* = try Matrix.initialize(allocator, parameter.value.row_count, parameter.value.column_count);

                @memcpy(x.entries, parameter.value.entries);
                @memcpy(z.entries, parameter.value.entries);
            }

            return ret;
        }

        pub fn updateParameters(self: *Self) void {
            for (self.parameters, self.x, self.z) |parameter, x, z| {
                for (parameter.value.entries, parameter.gradient.entries, x.entries, z.entries) |*y_value_entry, y_gradient_entry, *x_entry, *z_entry| {
                    y_value_entry.* = momentum_decay_coefficient * x_entry.* + (1 - momentum_decay_coefficient) * z_entry.*;
                    z_entry.* -= learning_rate * y_gradient_entry;
                    x_entry.* = (self.t / (self.t + 1)) * x_entry.* + (1 / (self.t + 1)) * z_entry.*;

                    self.t += 1;
                }
            }
        }
    };
}
