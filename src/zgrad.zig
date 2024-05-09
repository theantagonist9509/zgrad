pub const core = .{
    .Matrix = @import("matrix.zig").Matrix,
    .Program = @import("program.zig").Program,
};

pub const templates = .{
    .Mlp = @import("mlp.zig").Mlp,
};
