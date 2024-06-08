const std = @import("std");

pub const Matrix = struct {
    row_count: usize,
    column_count: usize,
    elements: []f32,

    pub fn initialize(allocator: std.mem.Allocator, row_count: usize, column_count: usize) !Matrix {
        return .{
            .row_count = row_count,
            .column_count = column_count,
            .elements = try allocator.alloc(f32, row_count * column_count),
        };
    }

    pub fn free(self: *Matrix, allocator: std.mem.Allocator) void {
        allocator.free(self.elements);
        self.elements.len = 0; // To prevent freed matrices from being serialized
    }

    pub fn copy(destination: Matrix, source: Matrix) void {
        @memcpy(destination.elements, source.elements);
    }

    pub fn getElement(self: Matrix, i: usize, j: usize, comptime transpose: bool) f32 {
        return self.elements[if (transpose) (j * self.column_count + i) else (i * self.column_count + j)];
    }

    pub fn getI(self: Matrix, index: usize) usize {
        return index / self.column_count;
    }

    pub fn getJ(self: Matrix, index: usize) usize {
        return index % self.column_count;
    }

    pub fn argmax(self: Matrix) usize {
        var maximum_index: usize = 0;

        for (self.elements[1..], 1..) |element, i| {
            if (element > self.elements[maximum_index])
                maximum_index = i;
        }

        return maximum_index;
    }

    pub fn add(out: Matrix, a: Matrix, b: Matrix) void {
        for (out.elements, a.elements, b.elements) |*element, a_element, b_element|
            element.* = a_element + b_element;

        // TODO implement matrix operations using simd
        //const out_vector = @as(@Vector(out.elements.len, f32), out.elements);
        //const a_vector = @as(@Vector(a.elements.len, f32), a.elements);
        //const b_vector = @as(@Vector(b.elements.len, f32), b.elements);
        //out_vector = a_vector + b_vector;
    }

    pub fn matrixMultiply(out: Matrix, a: Matrix, comptime transpose_a: bool, b: Matrix, comptime transpose_b: bool) void {
        @memset(out.elements, 0);

        for (out.elements, 0..) |*element, index| {
            const i = out.getI(index);
            const j = out.getJ(index);

            for (0..(if (transpose_a) a.row_count else a.column_count)) |k|
                element.* += a.getElement(i, k, transpose_a) * b.getElement(k, j, transpose_b);
        }
    }

    pub fn affineTransform(out: Matrix, a: Matrix, b: Matrix, c: Matrix) void {
        @memcpy(out.elements, c.elements);

        for (out.elements, 0..) |*element, index| {
            const i = out.getI(index);
            const j = out.getJ(index);

            for (0..a.column_count) |k| {
                element.* += a.getElement(i, k, false) * b.getElement(k, j, false);
                if (std.math.isNan(element.*)) {
                    std.debug.print("a_element: {}, b_element: {}\n", .{ a.getElement(i, k, false), b.getElement(k, j, false) });
                }
            }
        }
    }

    pub fn relu(out: Matrix, a: Matrix) void {
        for (out.elements, a.elements) |*element, a_element|
            element.* = if (a_element > 0) a_element else 0;
    }

    pub const leaky_relu_slope = 0.1;

    pub fn leakyRelu(out: Matrix, a: Matrix) void {
        for (out.elements, a.elements) |*element, a_element|
            element.* = if (a_element > 0) a_element else leaky_relu_slope * a_element;
    }

    pub fn sigmoid(out: Matrix, a: Matrix) void {
        for (out.elements, a.elements) |*element, a_element|
            element.* = 1 / (1 + @exp(-a_element));
    }

    pub fn softmax(out: Matrix, a: Matrix) void {
        var maximum = a.elements[0];

        for (a.elements[1..]) |element| {
            if (element > maximum)
                maximum = element;
        }

        var sum: f32 = 0;

        for (a.elements) |element|
            sum += @exp(element - maximum); // Biasing for numerical stability

        for (out.elements, a.elements) |*element, a_element|
            element.* = @exp(a_element - maximum) / sum;
    }

    //pub fn layerNormalization(out: Matrix, a: Matrix) void {
    //    var mean: f32 = 0;
    //    var sd: f32 = 0;

    //    for (a.elements) |element| {
    //        mean += element;
    //        sd += element * element;
    //    }

    //    mean /= a.elements.len;
    //    sd = @sqrt(sd / a.elements.len - mean * mean);

    //    for (out, a) |*element, a_element|
    //        element.* = (a_element - mean) / sd;
    //}

    pub fn meanSquaredError(out: Matrix, a: Matrix, b: Matrix) void { // TODO reverse operands?
        var sum: f32 = 0;

        for (a.elements, b.elements) |a_element, b_element|
            sum += (a_element - b_element) * (a_element - b_element);

        out.elements[0] = sum / @as(f32, @floatFromInt(a.elements.len));
    }

    pub fn crossEntropy(out: Matrix, a: Matrix, b: Matrix) void { // TODO reverse operands?
        out.elements[0] = 0;

        for (a.elements, b.elements) |a_element, b_element|
            out.elements[0] -= a_element * @log(b_element + std.math.floatEps(f32)); // Biasing for numerical stability
    }

    // See utilities.zig
    pub fn toggleSerializationFlag(self: *Matrix) void {
        self.elements.len = std.math.maxInt(usize) - self.elements.len;
    }
};
