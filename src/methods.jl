function doublebracket(
    operator::PauliSentence,
    ham,
    generator::PauliSentence,
)
    result = Dict{UInt,ComplexF64}()
    for (hkey, hvalue) in ham, (gkey, gvalue) in generator
        bg = gkey >> generator.qubits
        bh = hkey >> generator.qubits
        s1 = count_ones(bg & hkey)
        s2 = count_ones(bh & gkey)
        isodd(s1) ⊻ isodd(s2) || continue
        ghkey = gkey ⊻ hkey
        for (okey, ovalue) in operator
            bo = okey >> generator.qubits
            bhg = ghkey >> generator.qubits
            s3 = count_ones(bo & ghkey)
            s4 = count_ones(bhg & okey)
            isodd(s3) ⊻ isodd(s4) || continue
            key = okey ⊻ ghkey
            value = 4 * hvalue * gvalue * ovalue * (-1)^(s1 + s3)
            haskey(result, key) ? (result[key] += value) : (result[key] = value)
        end
    end
    return result
end

function doublebracket(
    operator::PauliSentence,
    ham::PauliSentence,
    generator::PauliSentence,
)
    chunks = Iterators.partition(ham, cld(length(ham), Threads.nthreads()))
    println(length(chunks))
    println(length(ham))
    tasks = map(chunks) do chunk
        Threads.@spawn doublebracket(operator, chunk, generator)
    end
    result = Dict{UInt,ComplexF64}()
    foreach(tasks) do task
        mergewith!(+, result, fetch(task))
    end
    return PauliSentence(result, operator.qubits, iscopy=false)
end

function dbf23(
    initial_hamiltonian::PauliSentence,
    generator::PauliSentence,
    interval::Tuple{<:Real,<:Real};
    rtol::Real=1e-3,
    atol::Real=1e-6,
    initial_h::Real=0.01,
)

    ham_norm = norm(values(initial_hamiltonian))
    x = [copy(initial_hamiltonian)]
    t = [interval[1]]
    h = initial_h
    k1 = doublebracket(initial_hamiltonian, initial_hamiltonian, generator)
    k1_res = copy(k1)
    y = copy(initial_hamiltonian)
    # iter = 0
    while t[end] < interval[2]

        # iter += 1
        map!(v -> ode23_a21 * h * v, values(k1))
        mergewith!(+, y, k1)
        k2 = doublebracket(y, y, generator)

        map!(v -> ode23_α * v, values(k1))
        map!(v -> ode23_a32 * h * v, values(k2))
        mergewith!(+, y, k1, k2)
        k3 = doublebracket(y, y, generator)

        map!(v -> ode23_β1 * v, values(k1))
        map!(v -> ode23_β2 * v, values(k2))
        map!(v -> ode23_b3 * h * v, values(k3))
        y_new = PauliSentence(mergewith(+, y, k1, k2, k3), y.qubits, iscopy=false)

        map!(v -> ode23_γ1 * v, values(k1))
        map!(v -> ode23_γ2 * v, values(k2))
        map!(v -> ode23_γ3 * v, values(k3))
        map!(v -> ode23_q4 * h * v, values(k3))
        mergewith!(+, y, k1, k2, k3)

        s = 0
        for (key, value) in y_new
            temp = abs(value - y[key]) / (atol + rtol * abs(value))
            temp > s && (s = temp)
        end
        if s < 1
            push!(t, t[end] + h)
            map!(v -> ham_norm * v / norm(values(y_new)), values(y_new))
            push!(x, y_new)
            k1 = k3
            k1_res = copy(k1)
            h *= min(5.0, 0.9 * s^(-1 / 3))
        else
            k1 = copy(k1_res)
            h *= max(0.1, 0.9 * s^(-0.5))
        end
        y = copy(x[end])
    end
    return x, t
end

function dbf45(
    initial_hamiltonian::PauliSentence,
    generator::PauliSentence,
    interval::Tuple{<:Real,<:Real};
    rtol::Real=1e-3,
    atol::Real=1e-6,
    initial_h::Real=0.01,
)

    ham_norm = norm(values(initial_hamiltonian))
    x = [copy(initial_hamiltonian)]
    t = [interval[1]]
    h = initial_h
    k1 = doublebracket(initial_hamiltonian, initial_hamiltonian, generator)
    k1_res = copy(k1)
    y = copy(initial_hamiltonian)
    # iter = 0
    while t[end] < interval[2]

        # iter += 1
        map!(v -> ode45_a21 * h * v, values(k1))
        mergewith!(+, y, k1)
        k2 = doublebracket(y, y, generator)

        map!(v -> ode45_α * v, values(k1))
        map!(v -> ode45_a32 * h * v, values(k2))
        mergewith!(+, y, k1, k2)
        k3 = doublebracket(y, y, generator)

        map!(v -> ode45_β1 * v, values(k1))
        map!(v -> ode45_β2 * v, values(k2))
        map!(v -> ode45_a43 * h * v, values(k3))
        mergewith!(+, y, k1, k2, k3)
        k4 = doublebracket(y, y, generator)

        map!(v -> ode45_γ1 * v, values(k1))
        map!(v -> ode45_γ2 * v, values(k2))
        map!(v -> ode45_γ3 * v, values(k3))
        map!(v -> ode45_a54 * h * v, values(k4))
        mergewith!(+, y, k1, k2, k3, k4)
        k5 = doublebracket(y, y, generator)

        map!(v -> ode45_δ1 * v, values(k1))
        map!(v -> ode45_δ2 * v, values(k2))
        map!(v -> ode45_δ3 * v, values(k3))
        map!(v -> ode45_δ4 * v, values(k4))
        map!(v -> ode45_a65 * h * v, values(k5))
        mergewith!(+, y, k1, k2, k3, k4, k5)
        k6 = doublebracket(y, y, generator)

        map!(v -> ode45_ε1 * v, values(k1))
        map!(v -> ode45_ε2 * v, values(k2))
        map!(v -> ode45_ε3 * v, values(k3))
        map!(v -> ode45_ε4 * v, values(k4))
        map!(v -> ode45_ε5 * v, values(k5))
        map!(v -> ode45_b6 * h * v, values(k6))
        y_new =
            PauliSentence(mergewith(+, y, k1, k2, k3, k4, k5, k6), y.qubits, iscopy=false)
        k7 = doublebracket(y_new, y_new, generator)

        map!(v -> ode45_ζ1 * v, values(k1))
        map!(v -> ode45_ζ2 * v, values(k2))
        map!(v -> ode45_ζ3 * v, values(k3))
        map!(v -> ode45_ζ4 * v, values(k4))
        map!(v -> ode45_ζ5 * v, values(k5))
        map!(v -> ode45_ζ6 * v, values(k6))
        map!(v -> ode45_q7 * h * v, values(k7))
        mergewith!(+, y, k1, k3, k4, k5, k6, k7)


        s = 0
        for (key, value) in y_new
            temp = abs(value - y[key]) / (atol + rtol * abs(value))
            temp > s && (s = temp)
        end
        if s < 1
            push!(t, t[end] + h)
            map!(v -> ham_norm * v / norm(values(y_new)), values(y_new))
            push!(x, y_new)
            k1 = k7
            k1_res = copy(k1)
            h *= min(5.0, 0.9 * s^(-0.2))
        else
            k1 = copy(k1_res)
            h *= max(0.1, 0.9 * s^(-0.25))
        end
        y = copy(x[end])
    end
    return x, t
end

function dbf23(
    initial_operator::PauliSentence,
    initial_hamiltonian::PauliSentence,
    generator::PauliSentence,
    interval::Tuple{<:Real,<:Real};
    rtol::Real=1e-3,
    atol::Real=1e-6,
    initial_h::Real=0.01,
)

    op_norm = norm(values(initial_operator))
    ham_norm = norm(values(initial_hamiltonian))
    x = [copy(initial_hamiltonian)]
    v = [copy(initial_operator)]
    t = [interval[1]]
    h = initial_h
    k1_ham = doublebracket(initial_hamiltonian, initial_hamiltonian, generator)
    k1_op = doublebracket(initial_operator, initial_hamiltonian, generator)
    k1_ham_res = copy(k1_ham)
    k1_op_res = copy(k1_op)
    y_ham = copy(initial_hamiltonian)
    y_op = copy(initial_operator)
    # iter = 0
    while t[end] < interval[2]
        # iter += 1
        map!(v -> ode23_a21 * h * v, values(k1_ham))
        mergewith!(+, y_ham, k1_ham)
        k2_ham = doublebracket(y_ham, y_ham, generator)
        map!(v -> ode23_a21 * h * v, values(k1_op))
        mergewith!(+, y_op, k1_op)
        k2_op = doublebracket(y_op, y_ham, generator)

        map!(v -> ode23_α * v, values(k1_ham))
        map!(v -> ode23_a32 * h * v, values(k2_ham))
        mergewith!(+, y_ham, k1_ham, k2_ham)
        k3_ham = doublebracket(y_ham, y_ham, generator)
        map!(v -> ode23_α * v, values(k1_op))
        map!(v -> ode23_a32 * h * v, values(k2_op))
        mergewith!(+, y_op, k1_op, k2_op)
        k3_op = doublebracket(y_op, y_ham, generator)

        map!(v -> ode23_β1 * v, values(k1_ham))
        map!(v -> ode23_β2 * v, values(k2_ham))
        map!(v -> ode23_b3 * h * v, values(k3_ham))
        y_ham_new = PauliSentence(
            mergewith(+, y_ham, k1_ham, k2_ham, k3_ham),
            y_ham.qubits,
            iscopy=false,
        )
        map!(v -> ode23_β1 * v, values(k1_op))
        map!(v -> ode23_β2 * v, values(k2_op))
        map!(v -> ode23_b3 * h * v, values(k3_op))
        y_op_new = PauliSentence(
            mergewith(+, y_op, k1_op, k2_op, k3_op),
            y_op.qubits,
            iscopy=false,
        )

        map!(v -> ode23_γ1 * v, values(k1_ham))
        map!(v -> ode23_γ2 * v, values(k2_ham))
        map!(v -> ode23_γ3 * v, values(k3_ham))
        map!(v -> ode23_q4 * h * v, values(k3_ham))
        mergewith!(+, y_ham, k1_ham, k2_ham, k3_ham)
        map!(v -> ode23_γ1 * v, values(k1_op))
        map!(v -> ode23_γ2 * v, values(k2_op))
        map!(v -> ode23_γ3 * v, values(k3_op))
        map!(v -> ode23_q4 * h * v, values(k3_op))
        mergewith!(+, y_op, k1_op, k2_op, k3_op)

        s = 0
        for (key, value) in y_ham_new
            temp = abs(value - y_ham[key]) / (atol + rtol * abs(value))
            temp > s && (s = temp)
        end
        for (key, value) in y_op_new
            temp = abs(value - y_op[key]) / (atol + rtol * abs(value))
            temp > s && (s = temp)
        end
        if s < 1
            push!(t, t[end] + h)
            map!(v -> ham_norm * v / norm(values(y_ham_new)), values(y_ham_new))
            push!(x, y_ham_new)
            k1_ham = k3_ham
            k1_ham_res = copy(k1_ham)
            map!(v -> op_norm * v / norm(values(y_op_new)), values(y_op_new))
            push!(v, y_op_new)
            k1_op = k3_op
            k1_op_res = copy(k1_op)
            h *= min(5.0, 0.9 * s^(-1 / 3))
        else
            k1_ham = copy(k1_ham_res)
            k1_op = copy(k1_op_res)
            h *= max(0.1, 0.9 * s^(-0.5))
        end
        y_ham = copy(x[end])
        y_op = copy(v[end])
    end
    return x, v, t
end

function dbf45(
    initial_operator::PauliSentence,
    initial_hamiltonian::PauliSentence,
    generator::PauliSentence,
    interval::Tuple{<:Real,<:Real};
    rtol::Real=1e-3,
    atol::Real=1e-6,
    initial_h::Real=0.01,
)

    op_norm = norm(values(initial_operator))
    ham_norm = norm(values(initial_hamiltonian))
    x = [copy(initial_hamiltonian)]
    v = [copy(initial_operator)]
    t = [interval[1]]
    h = initial_h
    k1_ham = doublebracket(initial_hamiltonian, initial_hamiltonian, generator)
    k1_op = doublebracket(initial_operator, initial_hamiltonian, generator)
    k1_ham_res = copy(k1_ham)
    k1_op_res = copy(k1_op)
    y_ham = copy(initial_hamiltonian)
    y_op = copy(initial_operator)
    # iter = 0
    while t[end] < interval[2]
        # iter += 1
        map!(v -> ode45_a21 * h * v, values(k1_ham))
        mergewith!(+, y_ham, k1_ham)
        k2_ham = doublebracket(y_ham, y_ham, generator)
        map!(v -> ode45_a21 * h * v, values(k1_op))
        mergewith!(+, y_op, k1_op)
        k2_op = doublebracket(y_op, y_ham, generator)

        map!(v -> ode45_α * v, values(k1_ham))
        map!(v -> ode45_a32 * h * v, values(k2_ham))
        mergewith!(+, y_ham, k1_ham, k2_ham)
        k3_ham = doublebracket(y_ham, y_ham, generator)
        map!(v -> ode45_α * v, values(k1_op))
        map!(v -> ode45_a32 * h * v, values(k2_op))
        mergewith!(+, y_op, k1_op, k2_op)
        k3_op = doublebracket(y_op, y_ham, generator)

        map!(v -> ode45_β1 * v, values(k1_ham))
        map!(v -> ode45_β2 * v, values(k2_ham))
        map!(v -> ode45_a43 * h * v, values(k3_ham))
        mergewith!(+, y_ham, k1_ham, k2_ham, k3_ham)
        k4_ham = doublebracket(y_ham, y_ham, generator)
        map!(v -> ode45_β1 * v, values(k1_op))
        map!(v -> ode45_β2 * v, values(k2_op))
        map!(v -> ode45_a43 * h * v, values(k3_op))
        mergewith!(+, y_op, k1_op, k2_op, k3_op)
        k4_op = doublebracket(y_op, y_ham, generator)

        map!(v -> ode45_γ1 * v, values(k1_ham))
        map!(v -> ode45_γ2 * v, values(k2_ham))
        map!(v -> ode45_γ3 * v, values(k3_ham))
        map!(v -> ode45_a54 * h * v, values(k4_ham))
        mergewith!(+, y_ham, k1_ham, k2_ham, k3_ham, k4_ham)
        k5_ham = doublebracket(y_ham, y_ham, generator)
        map!(v -> ode45_γ1 * v, values(k1_op))
        map!(v -> ode45_γ2 * v, values(k2_op))
        map!(v -> ode45_γ3 * v, values(k3_op))
        map!(v -> ode45_a54 * h * v, values(k4_op))
        mergewith!(+, y_op, k1_op, k2_op, k3_op, k4_op)
        k5_op = doublebracket(y_op, y_ham, generator)

        map!(v -> ode45_δ1 * v, values(k1_ham))
        map!(v -> ode45_δ2 * v, values(k2_ham))
        map!(v -> ode45_δ3 * v, values(k3_ham))
        map!(v -> ode45_δ4 * v, values(k4_ham))
        map!(v -> ode45_a65 * h * v, values(k5_ham))
        mergewith!(+, y_ham, k1_ham, k2_ham, k3_ham, k4_ham, k5_ham)
        k6_ham = doublebracket(y_ham, y_ham, generator)
        map!(v -> ode45_δ1 * v, values(k1_op))
        map!(v -> ode45_δ2 * v, values(k2_op))
        map!(v -> ode45_δ3 * v, values(k3_op))
        map!(v -> ode45_δ4 * v, values(k4_op))
        map!(v -> ode45_a65 * h * v, values(k5_op))
        mergewith!(+, y_op, k1_op, k2_op, k3_op, k4_op, k5_op)
        k6_op = doublebracket(y_op, y_ham, generator)

        map!(v -> ode45_ε1 * v, values(k1_ham))
        map!(v -> ode45_ε2 * v, values(k2_ham))
        map!(v -> ode45_ε3 * v, values(k3_ham))
        map!(v -> ode45_ε4 * v, values(k4_ham))
        map!(v -> ode45_ε5 * v, values(k5_ham))
        map!(v -> ode45_b6 * h * v, values(k6_ham))
        y_ham_new = PauliSentence(
            mergewith(+, y_ham, k1_ham, k2_ham, k3_ham, k4_ham, k5_ham, k6_ham),
            y_ham.qubits,
            iscopy=false,
        )
        k7_ham = doublebracket(y_ham_new, y_ham_new, generator)
        map!(v -> ode45_ε1 * v, values(k1_op))
        map!(v -> ode45_ε2 * v, values(k2_op))
        map!(v -> ode45_ε3 * v, values(k3_op))
        map!(v -> ode45_ε4 * v, values(k4_op))
        map!(v -> ode45_ε5 * v, values(k5_op))
        map!(v -> ode45_b6 * h * v, values(k6_op))
        y_op_new = PauliSentence(
            mergewith(+, y_op, k1_op, k2_op, k3_op, k4_op, k5_op, k6_op),
            y_op.qubits,
            iscopy=false,
        )
        k7_op = doublebracket(y_op_new, y_ham_new, generator)

        map!(v -> ode45_ζ1 * v, values(k1_ham))
        map!(v -> ode45_ζ2 * v, values(k2_ham))
        map!(v -> ode45_ζ3 * v, values(k3_ham))
        map!(v -> ode45_ζ4 * v, values(k4_ham))
        map!(v -> ode45_ζ5 * v, values(k5_ham))
        map!(v -> ode45_ζ6 * v, values(k6_ham))
        map!(v -> ode45_q7 * h * v, values(k7_ham))
        mergewith!(+, y_ham, k1_ham, k3_ham, k4_ham, k5_ham, k6_ham, k7_ham)
        map!(v -> ode45_ζ1 * v, values(k1_op))
        map!(v -> ode45_ζ2 * v, values(k2_op))
        map!(v -> ode45_ζ3 * v, values(k3_op))
        map!(v -> ode45_ζ4 * v, values(k4_op))
        map!(v -> ode45_ζ5 * v, values(k5_op))
        map!(v -> ode45_ζ6 * v, values(k6_op))
        map!(v -> ode45_q7 * h * v, values(k7_op))
        mergewith!(+, y_op, k1_op, k3_op, k4_op, k5_op, k6_op, k7_op)

        s = 0
        for (key, value) in y_ham_new
            temp = abs(value - y_ham[key]) / (atol + rtol * abs(value))
            temp > s && (s = temp)
        end
        for (key, value) in y_op_new
            temp = abs(value - y_op[key]) / (atol + rtol * abs(value))
            temp > s && (s = temp)
        end
        if s < 1
            push!(t, t[end] + h)
            map!(v -> ham_norm * v / norm(values(y_ham_new)), values(y_ham_new))
            map!(v -> op_norm * v / norm(values(y_op_new)), values(y_op_new))
            push!(v, y_op_new)
            push!(x, y_ham_new)
            k1_ham = k7_ham
            k1_ham_res = copy(k1_ham)
            k1_op = k7_op
            k1_op_res = copy(k1_op)
            h *= min(5.0, 0.9 * s^(-0.2))
        else
            k1_ham = copy(k1_ham_res)
            k1_op = copy(k1_op_res)
            h *= max(0.1, 0.9 * s^(-0.25))
        end
        y_ham = copy(x[end])
        y_op = copy(v[end])
    end
    return x, v, t
end
