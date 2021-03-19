#!/usr/bin/env python
# coding: utf-8

from sympy import *
init_printing(use_latex='mathjax')

a,b,c,d,e,f,g,h,x,y,x,z,u,w,k,t = symbols('a b c d e f g h x y x z u w k t')
α,β,γ,δ,θ,φ = symbols('alpha beta gamma delta theta varphi')
kα,kβ,kγ,kδ,kθ,kφ = symbols('k_{\\alpha} k_{\\beta} k_{\\gamma} k_{\\delta} k_{\\theta} k_{\\varphi}')
kx,ky,kz = symbols('k_x k_y k_z')
Ck = {α:kα,β:kβ,γ:kγ,δ:kδ,θ:kθ,φ:kφ,x:kx,y:ky,z:kz}

θ1,θ2,θ3 = symbols('theta_1 theta_2 theta_3')
x1,x2,y1,y2,z1,z2 = symbols('x_1 x_2 y_1 y_2 z_1 z_2')
αt,βt,γt,δt,θt,φt = symbols('\\dot{\\alpha} \\dot{\\beta} \\dot{\\gamma} \\dot{\\delta} \\dot{\\theta} \\dot{\\varphi}')
θ1t,θ2t,θ3t = symbols('\\dot{\\theta_1} \\dot{\\theta_2} \\dot{\\theta_3}')
x1t,x2t,y1t,y2t,z1t,z2t = symbols('\\dot{x_1} \\dot{x_2} \\dot{y_1} \\dot{y_2} \\dot{z_1} \\dot{z_2}')
xt,yt,zt = symbols('\\dot{x} \\dot{y} \\dot{z}')
Ve = {α:αt,β:βt,γ:γt,δ:δt,θ:θt,θ1:θ1t,θ2:θ2t,θ3:θ3t,φ:φt,x:xt,y:yt,z:zt,x1:x1t,x2:x2t,y1:y1t,y2:y2t,z1:z1t,z2:z2t}

def MecSolve_OneDOF(Cg,Eq):      # Cg,Eq Coords generalizados e Eq Restrição
    Eq = Matrix(Eq)
    Cs = [ i for i in Cg[1:] ]   # Lista com as coords secundárias
    J = Eq.jacobian(Cs)          # Jacobiano do sistema
    F = Eq.jacobian([Cg[0]])     # Obtenção da matriz F
    K = simplify(-(J**-1)*F)     # Obtenção da matriz K
    Ks = Matrix([ Ck[i] for i in Cg[1:] ])
    L = simplify( K.jacobian([Cg[0]]) + K.jacobian(Cs)*Ks )
    return(F, J, K, L)

def MecSolve_MultDOF(Cg,Eq):    # Cg,Vg Coords e Veloc Generalizadas
    f = len(Cg)-len(Eq)
    Eq = Matrix(Eq)
    J = Eq.jacobian(Cg[f:])     # Jacobiano do sistema
    F = Eq.jacobian([Cg[:f]])   # Obtenção da matriz F
    K = simplify(-(J**-1)*F)    # Obtenção da matriz K

    L = Matrix([])
    Tg = [ i**t for i in Cg ]
    Tg = [ Ve[i] for i in Cg ]
    for i in range(f):          # Obtenção da matriz L
        Lprov = K.col(i).jacobian(Cg[:f])*((1/Tg[i])*Matrix(Tg[:f])) + K.col(i).jacobian(Cg[f:])*((1/Tg[i])*Matrix(Tg[f:]))
        L = L.row_join(Lprov)
    return(F, J, K, L)