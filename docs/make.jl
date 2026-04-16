using Documenter
using Hydrodynamics

DocMeta.setdocmeta!(Hydrodynamics, :DocTestSetup, :(using Hydrodynamics); recursive = true)

makedocs(
    sitename = "Hydrodynamics.jl",
    modules = [Hydrodynamics],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(
    repo = "github.com/JuliaOceanWaves/Hydrodynamics.jl.git",
    devbranch = "main",
    push_preview = true
)
