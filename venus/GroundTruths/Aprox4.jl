include("funciones.jl")

#---------------------------------------- Carga de recortes -----------------------------------------------------------

# Carga de Hit 'n Miss

# TODO Aqui habria que meter los recortes de la aprox3 (los de dividir la imagen en 4)
hit = loadFolderImages(hit_path2)
#hit = positive_images
#Aqui habria que cargar las miss sin recortar
miss = loadFolderImages(miss_path3)
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

dataSetAux = vcat(first_part, second_part)




#---------------------------------------- Carga de recortes -----------------------------------------------------------

# Carga de Hit 'n Miss

# Aqui habria que meter los recortes normales (grandes)
hit = loadFolderImages(hit_path)
#hit = positive_images

miss = loadFolderImages(miss_path)
#miss = negative_images

# TODO Aqui habria que meter los recortes del agujero pequeño (pequeños)

hit1 = loadFolderImages(hit_path1) 

miss1 = loadFolderImages(miss_path1)

#---------------------------------- Extraccion de caracteristicas -----------------------------------------------------

first_part = []
second_part = []
third_part = []
fourth_part = []


for image in hit
    temp = featureExtraction(image, 0, [0, 1])
    temp = temp[1:2]
    push!(first_part, temp)
end

for image in miss
    temp = featureExtraction(image, 1, [0, 1])
    temp = temp[1:2]
    push!(second_part, temp)
end

for image in hit1 #Cargo los patrones positivos recortados mas pequeños
    push!(third_part, featureExtraction(image, 0, [0, 1]))
end

for image in miss1#Cargo los patrones negativos recortados mas pequeños
    push!(fourth_part, featureExtraction(image, 1, [0, 1]))
end

dataSet1 = vcat(first_part, second_part)
dataSet2 = vcat(third_part, fourth_part)

dataSet = [] #Ahora mismo es el dataset de la aprox2
i = 1
for fila in dataSet2
    new_fila = append!(dataSet1[i], fila)
    push!(dataSet, new_fila)
    global i = i+1
end



dataSet2 = []
i = 1
for fila in dataSet
    append!(dataSetAux[i], fila)
    #push!(dataSet2, new_fila)
    global i = i+1
end

# Guardamos el dataSet en "aprox3.data"

saveAsData(path*"/"*"aprox4.data", dataSetAux, '\t')

