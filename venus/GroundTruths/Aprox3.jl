include("funciones.jl")

#---------------------------------------- Carga de recortes -----------------------------------------------------------

# Carga de Hit 'n Miss

hit = loadFolderImages(hit_path2)
#hit = positive_images

miss = loadFolderImages(miss_path2)
#miss = negative_images


#---------------------------------- Extraccion de caracteristicas -----------------------------------------------------

first_part = []
cont = 1
array = []

for image in hit
    println("Cont:")
    println(cont)
    if(mod(cont,4) == 1)
        global array = []
    end
    element = featureExtraction(image, 0, [0, 1])
    global array = append!(array, element[1:2])

    if(mod(cont, 4) == 0)
        push!(first_part, array)
    end
    global cont = cont + 1
end

cont = 1
array = []
second_part = []

for image in miss
    println("Cont:")
    println(cont)
    if(mod(cont,4) == 1)
        global array = []
    end
    element = featureExtraction(image, 1, [0, 1])
    global array = append!(array, element[1:2])

    if(mod(cont, 4) == 0)
        push!(second_part, array)
    end
    global cont =cont + 1
end

dataSet1 = vcat(first_part, second_part)

println("Size: ")
println(size(dataSet1))
println(size(dataSet_aux))
dataSet = hcat(dataSet1, dataSet_aux)




#=
dataSet1 = []
cont = 1
array = []
for fila in dataSet
    if (mod(cont, 4) == 1)
        global array = [] #Cada vez que sea el primer elemento de los 4 se reinicializa el array
    end
    global array = append!(array, fila) #Para elementos 1 2 3 4 los metemos en el array

    if (mod(cont, 4) == 0) #Cuando tenemos los 4 elementos lo metemos en el dataSet
        push!(dataSet1, array)
    end
    global cont = cont + 1
end

filter!(n -> n == 1.0 || n == 0.0, dataSet1)

#normalizeMinMax!(dataSet[1])
=#
# Guardamos el dataSet en "aprox3.data"

saveAsData(path*"/"*"aprox3.data", dataSet, '\t')
