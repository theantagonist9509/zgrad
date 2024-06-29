const std = @import("std");

const Matrix = @import("matrix.zig");

const Symbol = @This();

value: Matrix,
gradient: Matrix,

pub fn initialize(allocator: std.mem.Allocator, row_count: usize, column_count: usize) !Symbol { // TODO comptime dimensions?
    return .{
        .value = try Matrix.initialize(allocator, row_count, column_count),
        .gradient = try Matrix.initialize(allocator, row_count, column_count),
    };
}
