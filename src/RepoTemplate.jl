module RepoTemplate

export greet

"""
    greet(name = "world") -> String

Return a friendly greeting.
"""
greet(name::AbstractString = "world") = "Hello, $(name)!"

end
