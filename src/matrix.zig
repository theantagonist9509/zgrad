// TODO implement matrix operations using simd

const std = @import("std");

const Matrix = @This();

row_count: usize,
column_count: usize,
entries: []f32,

pub fn initialize(allocator: std.mem.Allocator, row_count: usize, column_count: usize) !Matrix {
    return .{
        .row_count = row_count,
        .column_count = column_count,
        .entries = try allocator.alloc(f32, row_count * column_count),
    };
}

pub fn free(self: *Matrix, allocator: std.mem.Allocator) void {
    allocator.free(self.entries);
    self.entries.len = 0; // To prevent freed matrices from being serialized
}

pub fn getEntry(self: Matrix, i: usize, j: usize, comptime transpose: bool) f32 {
    return self.entries[if (transpose) (j * self.column_count + i) else (i * self.column_count + j)];
}

pub fn getI(self: Matrix, index: usize) usize { // rename to getRowIndex?
    return index / self.column_count;
}

pub fn getJ(self: Matrix, index: usize) usize { // rename to getColumnIndex?
    return index % self.column_count;
}

pub fn argmax(self: Matrix) usize {
    var maximum_index: usize = 0;

    for (self.entries[1..], 1..) |entry, i| {
        if (entry > self.entries[maximum_index])
            maximum_index = i;
    }

    return maximum_index;
}

pub fn accumulate(out: Matrix, a: Matrix) void {
    for (out.entries, a.entries) |*entry, a_entry|
        entry.* += a_entry;
}

pub fn hadamardMultiply(out: Matrix, a: Matrix, b: Matrix) void {
    for (out.entries, a.entries, b.entries) |*entry, a_entry, b_entry|
        entry.* = a_entry * b_entry;
}

pub fn matrixMultiply(out: Matrix, a: Matrix, comptime transpose_a: bool, b: Matrix, comptime transpose_b: bool) void {
    @memset(out.entries, 0);
    matrixMultiplyAccumulate(out, a, transpose_a, b, transpose_b);
}

// wx + b
pub fn affineTransform(out: Matrix, x: Matrix, w: Matrix, b: Matrix) void {
    @memcpy(out.entries, b.entries);
    matrixMultiplyAccumulate(out, w, false, x, false);
}

pub fn matrixMultiplyAccumulate(out: Matrix, a: Matrix, comptime transpose_a: bool, b: Matrix, comptime transpose_b: bool) void {
    for (out.entries, 0..) |*entry, index| {
        const i = out.getI(index);
        const j = out.getJ(index);

        for (0..(if (transpose_a) a.row_count else a.column_count)) |k|
            entry.* += a.getEntry(i, k, transpose_a) * b.getEntry(k, j, transpose_b);
    }
}

pub fn relu(out: Matrix, a: Matrix) void { // TODO change this stuff to return result and use it as needed in core.zig
    leakyRelu(0, out, a);
}

pub fn leakyRelu(slope: comptime_float, out: Matrix, a: Matrix) void {
    for (out.entries, a.entries) |*entry, a_entry|
        entry.* = if (a_entry > 0) a_entry else slope * a_entry;
}

pub fn tanh(out: Matrix, a: Matrix) void {
    for (out.entries, a.entries) |*entry, a_entry|
        entry.* = 1 - 2 / (1 + @exp(2 * a_entry));
}

pub fn sigmoid(out: Matrix, a: Matrix) void {
    for (out.entries, a.entries) |*entry, a_entry|
        entry.* = 1 / (1 + @exp(-a_entry));
}

pub fn softmax(out: Matrix, a: Matrix) void {
    var maximum = a.entries[0];

    for (a.entries[1..]) |entry| {
        if (entry > maximum)
            maximum = entry;
    }

    var sum: f32 = 0;

    for (a.entries) |entry|
        sum += @exp(entry - maximum); // Biasing for numerical stability

    for (out.entries, a.entries) |*entry, a_entry|
        entry.* = @exp(a_entry - maximum) / sum;
}

//pub fn layerNormalization(out: Matrix, a: Matrix) void {
//    var mean: f32 = 0;
//    var sd: f32 = 0;

//    for (a.entries) |entry| {
//        mean += entry;
//        sd += entry * entry;
//    }

//    mean /= a.entries.len;
//    sd = @sqrt(sd / a.entries.len - mean * mean);

//    for (out, a) |*entry, a_entry|
//        entry.* = (a_entry - mean) / sd;
//}

pub fn meanSquaredError(out: Matrix, y_hat: Matrix, y: Matrix) void { // TODO reverse operands?
    var sum: f32 = 0;

    for (y_hat.entries, y.entries) |y_hat_entry, y_entry|
        sum += (y_hat_entry - y_entry) * (y_hat_entry - y_entry);

    out.entries[0] = sum / @as(f32, @floatFromInt(y_hat.entries.len));
}

pub fn crossEntropy(out: Matrix, y_hat: Matrix, y: Matrix) void { // TODO reverse operands?
    out.entries[0] = 0;

    for (y_hat.entries, y.entries) |y_hat_entry, y_entry|
        out.entries[0] -= y_entry * @log(y_hat_entry + std.math.floatEps(f32)); // Biasing for numerical stability; TODO make bias larger than floatEps(f32)?
}

// See utilities.zig
pub fn toggleSerializationFlag(self: *Matrix) void {
    self.entries.len = std.math.maxInt(usize) - self.entries.len;
}
