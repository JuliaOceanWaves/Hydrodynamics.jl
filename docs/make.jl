using Documenter
using RepoTemplate

DocMeta.setdocmeta!(RepoTemplate, :DocTestSetup, :(using RepoTemplate); recursive=true)

makedocs(
    sitename="RepoTemplate.jl",
    modules=[RepoTemplate],
    format=Documenter.HTML(prettyurls=get(ENV, "CI", "false") == "true"),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(
    repo="github.com/JuliaOceanWaves/RepoTemplate.jl.git",
    devbranch="main",
    push_preview=true
)
