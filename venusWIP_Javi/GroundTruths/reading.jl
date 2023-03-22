using DelimitedFiles 

files_and_dirs = readdir(pwd())


dataset = readdlm("img1.lxyr")



function count()
    global num =2
    for i in files_and_dirs
        name = "img"*string(num)*".lxyr"
        if cmp(i,name)
            append!(dataset, readdlm(name)) 
            num +=1
        end
    end
end
count()
num -= 1

m = Array{T}(undef, num)
num = 1
function string_slice(data::{Array})
    for x in data
        m[num] = split(x, " ")
end

println(num)

dataset = readdlm("img1.lxyr", '\t') 
println(dataset) 
 
#= 
importamos con readdlm cada uno de los archivos .lxyr 
 
Para cada archivo, recorrerlo y eliminar las filas que tengan un 4 
=#