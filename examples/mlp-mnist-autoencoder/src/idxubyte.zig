const std = @import("std");

pub const IdxUbyte = struct {
    shape: []u32,
    data: []u8,

    pub fn initialize(allocator: std.mem.Allocator, file: std.fs.File) !IdxUbyte {
        var idx: IdxUbyte = undefined;
        var magic_number_bytes: [4]u8 = undefined;

        _ = try file.readAll(&magic_number_bytes); // Big endian

        try std.testing.expect(magic_number_bytes[0] == 0 and magic_number_bytes[1] == 0 and magic_number_bytes[2] == 8);

        const dimension_count = magic_number_bytes[3];
        idx.shape = try allocator.alloc(u32, dimension_count);

        var data_byte_count: usize = 1;
        for (idx.shape) |*dimension_size| {
            dimension_size.* = try file.reader().readInt(u32, .big);
            data_byte_count *= dimension_size.*;
        }

        idx.data = try allocator.alloc(u8, data_byte_count);

        _ = try file.readAll(idx.data);

        return idx;
    }

    pub fn free(self: IdxUbyte, allocator: std.mem.Allocator) void {
        allocator.free(self.shape);
        allocator.free(self.data);
    }
};
