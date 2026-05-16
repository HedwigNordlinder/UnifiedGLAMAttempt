if !isdefined(@__MODULE__, :SimpleProgress)
    mutable struct SimpleProgress
        label::String
        total::Int
        every::Int
        width::Int
        start_time::Float64
        last_update::Int
        enabled::Bool
    end

    function SimpleProgress(label::AbstractString, total::Integer;
                            enabled::Bool = false,
                            every::Integer = max(1, Int(total) ÷ 100),
                            width::Integer = 32)
        SimpleProgress(String(label), max(Int(total), 1), max(Int(every), 1),
                       max(Int(width), 8), time(), -typemax(Int), enabled)
    end

    function format_seconds(x::Real)
        !isfinite(x) && return "Inf"
        x < 60 && return string(round(x; digits = 1), "s")
        x < 3600 && return string(round(x / 60; digits = 1), "m")
        string(round(x / 3600; digits = 2), "h")
    end

    function progress_update!(prog::SimpleProgress, iter::Integer; force::Bool = false, suffix = "")
        prog.enabled || return nothing
        it = clamp(Int(iter), 0, prog.total)
        should_print = force || it == 0 || it == prog.total || it - prog.last_update >= prog.every
        should_print || return nothing

        frac = it / prog.total
        filled = clamp(floor(Int, prog.width * frac), 0, prog.width)
        elapsed = time() - prog.start_time
        rate = it > 0 ? it / max(elapsed, eps(Float64)) : 0.0
        eta = rate > 0 ? (prog.total - it) / rate : Inf
        bar = repeat("#", filled) * repeat(".", prog.width - filled)
        pct = lpad(string(round(100 * frac; digits = 1)), 5)
        extra = isempty(String(suffix)) ? "" : " " * String(suffix)
        print("\r$(prog.label) [$bar] $pct% $it/$(prog.total) elapsed=$(format_seconds(elapsed)) eta=$(format_seconds(eta))$extra")
        it == prog.total && println()
        flush(stdout)
        prog.last_update = it
        nothing
    end
end
