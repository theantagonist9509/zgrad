const std = @import("std");

// Slices with len > (std.math.maxInt(usize) / 2) will not have their contents serialized; only their memory occupancy will be taken note of (as having length (std.math.maxInt(usize) - len))
pub fn serializeObject(object: anytype, file: std.fs.File, pointer_list: *std.ArrayList(usize)) !void {
    const writer = file.writer();

    switch (@typeInfo(@TypeOf(object))) {
        .Struct => |struct_info| {
            inline for (struct_info.fields) |field_info|
                try serializeObject(@field(object, field_info.name), file, pointer_list);
        },
        .Pointer => |pointer_info| {
            switch (pointer_info.size) {
                .Slice => blk: {
                    for (pointer_list.items, 0..) |item, index| {
                        if (item == @intFromPtr(object.ptr)) {
                            try writer.writeInt(usize, index, .little);
                            try writer.writeInt(usize, object.len, .little);

                            break :blk;
                        }
                    }

                    try pointer_list.append(@intFromPtr(object.ptr));
                    try writer.writeInt(usize, pointer_list.items.len - 1, .little);
                    try writer.writeInt(usize, object.len, .little);

                    if (object.len > std.math.maxInt(usize) / 2)
                        break :blk;

                    switch (@typeInfo(pointer_info.child)) {
                        .Struct, .Pointer => {
                            for (object) |item|
                                try serializeObject(item, file, pointer_list);
                        },
                        else => try file.writeAll(bytesFromSlice(object)),
                    }
                },
                else => unreachable,
            }
        },
        else => try file.writeAll(std.mem.asBytes(&object)),
    }
}

pub fn deserializeObject(T: type, allocator: std.mem.Allocator, file: std.fs.File, pointer_list: *std.ArrayListUnmanaged(usize)) !T {
    var t: T = undefined;

    const reader = file.reader();

    switch (@typeInfo(T)) {
        .Struct => |struct_info| {
            inline for (struct_info.fields) |field_info|
                @field(t, field_info.name) = try deserializeObject(field_info.type, allocator, file, pointer_list);
        },
        .Pointer => |pointer_info| {
            switch (pointer_info.size) {
                .Slice => {
                    const pointer_index = try reader.readInt(usize, .little);
                    const serialized_length = try reader.readInt(usize, .little);

                    const length = if (serialized_length > std.math.maxInt(usize) / 2) (std.math.maxInt(usize) - serialized_length) else serialized_length;

                    if (pointer_index < pointer_list.items.len) {
                        t.ptr = @ptrFromInt(pointer_list.items[pointer_index]);
                        t.len = length;
                    } else {
                        t = try allocator.alloc(pointer_info.child, length);
                        try pointer_list.append(allocator, @intFromPtr(t.ptr));

                        if (length == serialized_length) {
                            switch (@typeInfo(pointer_info.child)) {
                                .Struct, .Pointer => {
                                    for (t) |*item_pointer|
                                        item_pointer.* = try deserializeObject(pointer_info.child, allocator, file, pointer_list);
                                },
                                else => _ = try file.readAll(bytesFromSlice(t)),
                            }
                        }
                    }
                },
                else => unreachable,
            }
        },
        else => _ = try file.readAll(std.mem.asBytes(&t)),
    }

    return t;
}

// Zig 0.12.0 does not implement @ptrCast between slices of different lengths :(
fn bytesFromSlice(slice: anytype) []u8 {
    var bytes: []u8 = undefined;
    bytes.ptr = @ptrCast(slice.ptr);
    bytes.len = slice.len * @sizeOf(@TypeOf(slice[0]));

    return bytes;
}
