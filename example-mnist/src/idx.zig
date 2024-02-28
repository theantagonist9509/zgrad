const std = @import("std");

pub const Idx = struct {
    shape: []u32,
    data: []u8,

    pub fn initialize(allocator: std.mem.Allocator, file: std.fs.File) !Idx {
        var idx: Idx = undefined;
        var magic_number_bytes: [4]u8 = undefined;

        _ = try file.readAll(&magic_number_bytes);

        std.debug.assert(magic_number_bytes[0] == 0 and magic_number_bytes[1] == 0 and magic_number_bytes[2] == 8);

        const dimension_count = magic_number_bytes[3];
        idx.shape = try allocator.alloc(u32, dimension_count);

        var data_byte_count: usize = 1;
        for (idx.shape) |*dimension_size| {
            _ = try file.readAll(@as(*[4]u8, @ptrCast(dimension_size)));
            dimension_size.* = @byteSwap(dimension_size.*);
            data_byte_count *= dimension_size.*;
        }

        idx.data = try allocator.alloc(u8, data_byte_count);

        _ = try file.readAll(idx.data);

        return idx;
    }

    pub fn free(self: Idx, allocator: std.mem.Allocator) void {
        allocator.free(self.shape);
        allocator.free(self.data);
    }
};
