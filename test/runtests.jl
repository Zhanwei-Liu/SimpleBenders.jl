using Test

import SimpleBenders
using JuMP

using HiGHS
using LinearAlgebra

solver = optimizer_with_attributes(HiGHS.Optimizer, "log_to_console" => false)
# solver = optimizer_with_attributes(Gurobi.Optimizer, "LogToConsole" => false)
# test from http://www.iems.ucf.edu/qzheng/grpmbr/seminar/Yuping_Intro_to_BendersDecomp.pdf

function test_data()
    c = [2., 3.]
    A = [1 2;2 -1]
    D = zeros(2, 1) .+ [1, 3]
    b = [3, 4]
    return SimpleBenders.SubProblemData(b, D, A, c)
end

function test_result()
    d = test_data()
    m = Model(solver)
    set_silent(m)
    @variable(m, x[1:2] >= 0)
    @variable(m, y[1:1] >= 0)
    @objective(m, Min, d.c'*x + 2y[1])
    @constraint(m, d.A * x + d.D * y .>= d.b)
    optimize!(m)
    return (JuMP.value.(x), JuMP.value.(y), JuMP.objective_value(m))
end

# add a test example for feasibility cut
# test from Ray (Jian) Zhang. Learn Optimization Visually: Benders Decomposition. url: https://www.youtube.com/watch?app=desktop&v=vQzpydNOWDY
function test_data_ex2()
    c = vec([-1-0.01*i for i in 1:10])
    A = vcat(fill(-1.0,(1,10)), -Matrix(I, 10, 10))
    D = vcat([-1.0], -zeros(10,1))
    b = vec(hcat([-1000.0], fill(-100.0,(1,10))))
    return SimpleBenders.SubProblemData(b, D, A, c)
end

function test_result_ex2()
    d = test_data_ex2()
    m = Model(solver)
    set_silent(m)
    @variable(m, x[1:10] >= 0)
    @variable(m, y[1:1] >= 0, integer=true)
    @objective(m, Min, d.c'*x - 1.045y[1])
    @constraint(m, d.A * x + d.D * y .>= d.b)
    optimize!(m)
    return (JuMP.value.(x), JuMP.value.(y), JuMP.objective_value(m))
end

# test from Illustrative Example 3.3 (Complicating variables: Dual problem)
function test_data_ex3()
    c = vec([4.0, 3, 2, 3, 2])
    A = - [1.0 2 0 0 0; 2 1 0 0 0; -2 3 0 0 0; 0 0 1 0 0; 0 0 2 0 0; 0 0 0 1 0; 0 0 0 2 4; 0 0 0 3 1]
    D = zeros(8, 1) .- [2.0, 1, 1, 3, -1, 0, 3, -1]
    b = - vec([3.0, 3, 7, 4, 3, 1, 5, 4])
    return SimpleBenders.SubProblemData(b, D, A, c)
end

function test_result_ex3()
    d = test_data_ex3()
    m = Model(solver)
    set_silent(m)
    @variable(m, x[1:5] >= 0)
    @variable(m, y[1:1] >= 0)
    @objective(m, Min, d.c'*x + 3y[1])
    @constraint(m, d.A * x + d.D * y .>= d.b)
    optimize!(m)
    return (JuMP.value.(x), JuMP.value.(y), JuMP.objective_value(m))
end

# Computational Example 3.1 (Subproblem Infeasibility)
function test_data_ex4()
    c = vec([-1/4.0])
    A = zeros(5, 1) .- [-1.0, -1/2, 1/2, 1, 1]
    D = zeros(5, 1) .- [1.0, 1, 1, -1, 0]
    b = - vec([5.0, 15/2, 35/2, 10, 16])
    return SimpleBenders.SubProblemData(b, D, A, c)
end

function test_result_ex4()
    d = test_data_ex4()
    m = Model(solver)
    set_silent(m)
    @variable(m, x[1:1] >= 0)
    @variable(m, y[1:1] >= 0)
    @objective(m, Min, d.c'*x - y[1])
    @constraint(m, d.A * x + d.D * y .>= d.b)
    optimize!(m)
    return (JuMP.value.(x), JuMP.value.(y), JuMP.objective_value(m))
end

# Computational Example 3.2 (Subproblem Infeasibility)
function test_data_ex5()
    c = vec([-2., -1., 1.])
    A = - [1. 0. 0.; 0. 2. 0.; 0. 0. 1.; 0. 0. 0.]
    D = - [1. 1.; 3. 0.; 0. -7.;  -1. 1.]
    b = - vec([3., 12., -16., 2.])
    return SimpleBenders.SubProblemData(b, D, A, c)
end

function test_result_ex5()
    d = test_data_ex5()
    m = Model(solver)
    set_silent(m)
    @variable(m, x[1:3] >= 0)
    @variable(m, y[1:2] >= 0)
    @objective(m, Min, d.c'*x + 3y[1] - 3y[2])
    @constraint(m, d.A * x + d.D * y .>= d.b)
    optimize!(m)
    return (JuMP.value.(x), JuMP.value.(y), JuMP.objective_value(m))
end

# Exercise 3.2
function test_data_ex6()
    c = vec([2., 2.5, .5, 4.])
    A = - [-2. 3. 0. 0.; 2. 4. 0. 0.; 0. 0. 2. -1.; 0. 0. -0.5 -1.]
    D = zeros(4,1) - [-4.; 1.; -1.; 3.]
    b = - vec([-4., 2.5, .5, -3.])
    return SimpleBenders.SubProblemData(b, D, A, c)
end

function test_result_ex6()
    d = test_data_ex6()
    m = Model(solver)
    set_silent(m)
    @variable(m, x[1:4] >= 0)
    @variable(m, y[1:1] >= 0)
    @objective(m, Min, d.c'*x + 3y[1])
    @constraint(m, d.A * x + d.D * y .>= d.b)
    optimize!(m)
    return (JuMP.value.(x), JuMP.value.(y), JuMP.objective_value(m))
end

# 1.4.1 Two-year Coal and Gas Procurement
function test_data_ex7()
    c = vec([2.25, 2.55, 3., 2.75, 0.6, 0.8])
    A = - [0.  0.  0.  0.  0.  0.; 
           1.  1.  0.  0.  0.  0.;
          -1. -1.  0.  0.  0.  0.;
           1.  0.  0.  0.  0.  0.;
          -1.  0.  0.  0.  0.  0.;
           0.  1.  0.  0.  0.  0.;
           0. -1.  0.  0.  0.  0.;
           0.  0.  1.  1.  0.  0.;
           0.  0. -1. -1.  0.  0.;
           0.  0.  1.  0.  0.  0.;
           0.  0. -1.  0.  0.  0.;
           0.  0.  0.  1.  0.  0.;
           0.  0.  0. -1.  0.  0.;
           0.  0.  0.  0.  1.  1.;
           0.  0.  0.  0. -1. -1.;
           0.  0.  0.  0.  1.  0.;
           0.  0.  0.  0. -1.  0.;
           0.  0.  0.  0.  0.  1.;
           0.  0.  0.  0.  0. -1.]
    D = - [-1. -1.;  1.  1.; -1. -1.;  1.  0.; -1.  0.;  0.  1.;  0. -1.; 1.  1.; -1. -1.; 1.  0.;
           -1.  0.;  0.  1.;  0. -1.;  1.  1.; -1. -1.;  1.  0.; -1.  0.; 0.  1.;  0. -1.;]
    b = - vec([-750., 1650., -1650., 1100., -550., 1100., -550., 1500., -1500., 1000., -500., 1000., -500., 1300., -1300., 866., -433., 866., -433.])
    return SimpleBenders.SubProblemData(b, D, A, c)
end

function test_result_ex7()
    d = test_data_ex7()
    m = Model(solver)
    set_silent(m)
    @variable(m, x[1:6] >= 0)
    @variable(m, y[1:2] >= 0)
    @objective(m, Min, d.c'*x + 4.5y[1] + 5.1y[2])
    @constraint(m, d.A * x + d.D * y .>= d.b)
    optimize!(m)
    return (JuMP.value.(x), JuMP.value.(y), JuMP.objective_value(m))
end


@testset "Test Optimility cut" begin
    data = test_data()
    f(v) = 2v[1]
    m = Model(solver)
    @variable(m, y[j=1:1] >= 0)
    (m, y, cuts, nopt_cons, nfeas_cons) = SimpleBenders.benders_optimize!(m, y, data, solver, f)
    (xref, yref, objref) = test_result()
    @test yref[1] ≈ JuMP.value(y[1])
    @test objref ≈ JuMP.objective_value(m)
end

@testset "Test mixed integer master problem" begin
    (xref, yref, objref) = test_result_ex2()
    data = test_data_ex2()
    f(v) = -1.045v[1]
    m = Model(solver)
    @variable(m, 0<=y[j=1:1] <= 1000, integer=true)
    (m, y, cuts, nopt_cons, nfeas_cons) = SimpleBenders.benders_optimize!(m, y, data, solver, f)
    
    @test yref[1] ≈ JuMP.value(y[1])
    @test objref ≈ JuMP.objective_value(m)
end

@testset "Test Feasibility cut" begin
    (xref, yref, objref) = test_result_ex3()
    data = test_data_ex3()
    f(v) = 3v[1]
    m = Model(solver)

    @variable(m, 0<=y[j=1:1] <= 3/2)
    (m, y, cuts, nopt_cons, nfeas_cons) = SimpleBenders.benders_optimize!(m, y, data, solver, f)
    
    @test yref[1] ≈ JuMP.value(y[1])
    @test objref ≈ JuMP.objective_value(m)
end 

@testset "Test Feasibility cut" begin
    (xref, yref, objref) = test_result_ex4()
    data = test_data_ex4()
    f(v) = -v[1]
    m = Model(solver)
    @variable(m, 0<=y[j=1:1] <= 35/2)
    (m, y, cuts, nopt_cons, nfeas_cons) = SimpleBenders.benders_optimize!(m, y, data, solver, f)
    
    @test yref[1] ≈ JuMP.value(y[1])
    @test objref ≈ JuMP.objective_value(m)
end 

@testset "Test multiple complex variables" begin
    (xref, yref, objref) = test_result_ex5()
    data = test_data_ex5()
    f(v) = 3v[1] - 3v[2]
    m = Model(solver)

    @variable(m, 0<=y[j=1:2]<=3)
    (m, y, cuts, nopt_cons, nfeas_cons) = SimpleBenders.benders_optimize!(m, y, data, solver, f)
    
    @test yref[1] ≈ JuMP.value(y[1])
    @test yref[2] ≈ JuMP.value(y[2])
    @test objref ≈ JuMP.objective_value(m)
end

@testset "Test example 6" begin
    (xref, yref, objref) = test_result_ex6()
    data = test_data_ex6()
    f(v) = 3v[1]
    m = Model(solver)

    @variable(m, 0<=y[j=1:1] <= 2.5)
    (m, y, cuts, nopt_cons, nfeas_cons) = SimpleBenders.benders_optimize!(m, y, data, solver, f)
    
    @test yref[1] ≈ JuMP.value(y[1])
    @test objref ≈ JuMP.objective_value(m)
end

@testset "Test example 7" begin
    (xref, yref, objref) = test_result_ex7()
    data = test_data_ex7()
    f(v) = 4.5v[1] + 5.1v[2]
    m = Model(solver)

    @variable(m, 0<=y[j=1:2]<=1300)
    (m, y, cuts, nopt_cons, nfeas_cons) = SimpleBenders.benders_optimize!(m, y, data, solver, f)
    @test yref[1] ≈ JuMP.value(y[1])
    @test yref[2] ≈ JuMP.value(y[2])
    @test objref ≈ JuMP.objective_value(m)
end



