using DelimitedFiles
using JLD2
using Images
using Plots
using Colors
using FileIO
using Images
using NaturalSort

include("funciones.jl")

# Debe ser común a todos los PC's
ccd("Venus_AA/venus/GroundTruths")

#Guardamos el directorio actual
files_and_dirs = readdir()
for f in files_and_dirs
    if filesize(f) < 3
        rm(f)
    end
end

#Retiramos del directorio actual los ficheros que no acaben en .lxyr
filter!(endswith(".lxyr"), files_and_dirs)

#Inicializamos la matriz con 1 fila y el numero de atributos de columnas
global dataset = []

#134 = Numero maximo que queremos leer
for count in 1:134
    name = "img"*string(count)*".lxyr"
    if isfile(name) #Checkeamos si existe
        matrix = readdlm(name)
        push!(dataset, matrix)
        #global dataset = vcat(dataset, matrix) #Concatenamos verticalmente la matriz con el dataset
    else
        push!(dataset, Float64[])
    end
    #println(count)
    count = count + 1
end
#Printeamos el dataset original
#println(dataset)


size_data = size(dataset, 1)

dataset_4 = []
#Ahora creamos una matriz que guarde lo que fijo son volcanes (4) y los que no (1)
for i=1:size_data
    aux = []
    for j=1:(size(dataset[i], 1))
        if (dataset[i][j, 1] == 4.0)
            push!(aux, dataset[i][j,:])
        end
    end
    push!(dataset_4, aux)
end

dataset_1 = []
#Ahora creamos una matriz que guarde lo que fijo son volcanes (4) y los que no (1)
for i=1:size_data
    aux = []
    for j=1:(size(dataset[i], 1))
        if (dataset[i][j, 1] == 1.0)
            push!(aux, dataset[i][j,:])
        end
    end
    push!(dataset_1, aux)
end
#dataset_4[i] for i=1:size_data  [dataset[i,:] for i=1:size_data if dataset[i,1]==4.0]
#dataset_1= [dataset[i,:] for i=1:size_data if dataset[i,1]==1.0]
#=
println(dataset_4) #Se maneja con [fila][columna].
println()
println(dataset_1) #Se maneja con [fila][columna].
println()
#Ahora mismo el dataset es una matriz con todos los patrones
=#
return

img_path = mycd*"Venus_AA/venus"
hit_path = mycd*"Venus_AA/venus/hit"
miss_path = mycd*"Venus_AA/venus/miss"

function count(dir::String)
    content = readdir(dir)
    filter!(endswith(".png"), content)
    global num =0
    for i in content
            num +=1
    end
end
#count(img_path)
#println(num)
function count(arr::Vector{Any})
    global num =0
    for i in arr
            num +=1
    end
end

function loadFolderImages(folderName::String)
    images = [];
    files = sort(readdir(folderName), lt=natural)
    filter!(endswith(".png"), files)
    println(files)
    for fileName in files
        image = load(img_path*"/"*fileName);
            
        # Check that they are color images
        #@assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
        # Add the image to the vector of images
        push!(images, convert(Array{Float64}, gray.(Gray.(image))));
    end;

    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors ||Solo nos interesa lo gris(imageToColorArray.(images), 
    return images;
end;

# Tiene todas las imagenes, 1 por fila 
matrix = loadFolderImages(img_path)

save("gray.png", colorview(Gray,matrix[1]))

#display(plot(matrix))

positive_images = []
negative_images = []

for i=1:size_data
    for j=1:(size(dataset_4[i], 1))
        println(dataset_4[i][j])
        recorte = dataset_4[i][j][2:4]
        push!(positive_images, recortar(recorte, matrix[i]))
    end
end

for i=1:size_data
    for j=1:(size(dataset_1[i], 1))
        println(dataset_1[i][j])
        recorte = dataset_1[i][j][2:4]
        push!(negative_images, recortar(recorte, matrix[i]))
    end
end

display(positive_images[1])

# Recorrer dataset_4
    # Para cada dataset_4[i], aplicarle a images[i] los recortes y apilarlos en positive_images[] de la siguiente manera push!(positive_images, recorte)

# matrix_c = matrix[1]

# coord = [convert(UInt16,dataset_4[1][2]), convert(UInt16,dataset_4[1][3]), convert(UInt16,round(dataset_4[1][4])) ] #x, y, radius

# matrix_cut = matrix_c[coord[1],coord[2]]

# println(matrix_cut)

# Crear el directorio si no existe
if !isdir("imagenes_positivas")
    mkdir("imagenes_positivas")
end
global cont = 1
for image in positive_images
# Guardar la imagen en el directorio
name = "recorte"*string(cont)*".png"
save(joinpath("imagenes_positivas", name), image)
global cont = cont+1
end

# Crear el directorio si no existe
if !isdir("imagenes_negativas")
    mkdir("imagenes_negativas")
end
global cont = 1
for image in negative_images
# Guardar la imagen en el directorio
name = "recorte"*string(cont)*".png"
save(joinpath("imagenes_negativas", name), image)
global cont = cont+1
end

