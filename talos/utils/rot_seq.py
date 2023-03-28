from csdl import einsum


def rot_seq(second, first):
    return einsum(second, first, subscripts='ijl,jkl->ikl')
