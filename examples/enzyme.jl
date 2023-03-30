using Enzyme

function f(x,y)
    y .= x .* x
    return nothing
end

x  = [2.0]
rx = [0.0]
y  = [0.0]
ry = [1.0]

# Reverse
autodiff(Reverse, f, Const, Duplicated(x, rx), Duplicated(y, ry))

x  = [2.0]
fx = [1.0]
y  = [0.0]
fy = [0.0]

# Forward
autodiff(Forward, f, Const, Duplicated(x, fx), Duplicated(y, fy))

@assert rx[1] == fy[1]

# Forward over Forward
x   = [2.0]
Fx  = [1.0]

fx  = [1.0]
Ffx = [0.0]


y  = [0.0]
Fy = [0.0]

fy  = [0.0]
Ffy = [0.0]

autodiff(Forward, (x, y) -> Enzyme.autodiff_deferred(
    Forward, f, Const, x, y), Const,
    Duplicated(Duplicated(x, Fx), Duplicated(fx, Ffx)),
    Duplicated(Duplicated(y, Fy), Duplicated(fy, Ffy)),
)

@assert Fy[1] == rx[1]
@assert fy[1] == rx[1]

# Reverse over Forward
x   = [2.0]
Fx  = [1.0]

rx  = [0.0]
Frx = [0.0]


y  = [0.0]
Fy = [0.0]

ry  = [1.0]
Fry = [0.0]

autodiff(Forward, (x, y) -> Enzyme.autodiff_deferred(
    Reverse, f, Const, x, y), Const,
    Duplicated(Duplicated(x, rx), Duplicated(Fx, Frx)),
    Duplicated(Duplicated(y, ry), Duplicated(Fy, Fry)),
)

@assert Frx[1] == Ffy[1]
@assert fy[1] == Fy[1]
@assert fy[1] == rx[1]