include("personal.jl")

function ccd(relativeLocation::String)
    cd(mycd*relativeLocation)
end