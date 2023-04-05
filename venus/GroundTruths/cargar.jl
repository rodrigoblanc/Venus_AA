# Import packages
#import Pkg; Pkg.add("JLD2")
#import Pkg; Pkg.add("Images")
#import Pkg; Pkg.add("Plots")

using DelimitedFiles
using JLD2
using Images
using Plots
using Colors
using FileIO
using Images
using NaturalSort
using Images
using Statistics
include("funciones.jl")

# PATHS

pattern_path = mycd*"Venus_AA/venus/GroundTruths/Patrones"
img_path = mycd*"Venus_AA/venus/imagenes"
hit_path = mycd*"Venus_AA/venus/hit"
miss_path = mycd*"Venus_AA/venus/miss"


# Args

# Número de imágenes a procesar
numberOfImages = 134

for i=1:(size(ARGS,1))
    if ARGS[i] == "-numImages"
        try
            numberOfImages = ARGS[i+1]
        catch
            error("No argument left but expected the number of images to process")
        end
    end
end 
# Debe ser común a todos los PC's
#ccd("Venus_AA/venus/GroundTruths/dataset_etiquetado")

#Guardamos el directorio actual
files_and_dirs = readdir(pattern_path)
println(files_and_dirs)
for f in files_and_dirs
    if filesize(pattern_path*"/"*f) < 3
        rm(pattern_path*"/"*f)
    end
end

#Retiramos del directorio actual los ficheros que no acaben en .lxyr
filter!(endswith(".lxyr"), files_and_dirs)

#Inicializamos la matriz con 1 fila y el numero de atributos de columnas
global dataset = []

#134 = Numero maximo que queremos leer
for count in 1:numberOfImages
    name = "img"*string(count)*".lxyr"
    if isfile(pattern_path*"/"*name) #Checkeamos si existe
        matrix = readdlm(pattern_path*"/"*name)
        push!(dataset, matrix)
        #global dataset = vcat(dataset, matrix) #Concatenamos verticalmente la matriz con el dataset
    else
        push!(dataset, Float64[])
    end
    #println(count)
    count = count + 1
end
#Printeamos el dataset original
println(dataset[1])


size_data = size(dataset, 1)

negative_dataset = []
#Ahora creamos una matriz que guarde lo que fijo son volcanes (1) y los que no (4)
for i=1:size_data
    aux = []
    for j=1:(size(dataset[i], 1))
        if (dataset[i][j, 1] == 4.0)
            push!(aux, dataset[i][j,:])
        end
    end
    push!(negative_dataset, aux)
end

println("Size data negative "*string(size(negative_dataset, 1)))

positive_dataset = []
#Ahora creamos una matriz que guarde lo que fijo son volcanes (1) y los que no (4)
for i=1:size_data
    aux = []
    for j=1:(size(dataset[i], 1))
        if (dataset[i][j, 1] == 1.0)
            push!(aux, dataset[i][j,:])
        end
    end
    push!(positive_dataset, aux)
end
#negative_dataset[i] for i=1:size_data  [dataset[i,:] for i=1:size_data if dataset[i,1]==4.0]
#positive_dataset= [dataset[i,:] for i=1:size_data if dataset[i,1]==1.0]
#=
println(negative_dataset) #Se maneja con [fila][columna].
println()
println(positive_dataset) #Se maneja con [fila][columna].
println()
#Ahora mismo el dataset es una matriz con todos los patrones
=#
return

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
        println("Loading filename: "*fileName)
        image = load(img_path*"/"*fileName);
            
        # Check that they are color images
        #@assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
        # Add the image to the vector of images
        push!(images, (fileName, convert(Array{Float64}, gray.(Gray.(image)))));
    end;

    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors ||Solo nos interesa lo gris(imageToColorArray.(images), 
    return images;
end;

# Tiene todas las imagenes, 1 por fila 
matrix = loadFolderImages(img_path)

save("gray.png", colorview(Gray,matrix[1][2]))

#display(plot(matrix))

positive_images = []
negative_images = []

for i=1:size_data
    println("Hola"*string(i))
    for j=1:(size(negative_dataset[i], 1))
        println(negative_dataset[i][j])
        println("on "*(matrix[i][1]))
        recorte = negative_dataset[i][j][2:4]
        push!(negative_images, (matrix[i][1], recortar(recorte, matrix[i][2])))
    end
end

for i=1:size_data
    for j=1:(size(positive_dataset[i], 1))
        println(positive_dataset[i][j])
        recorte = positive_dataset[i][j][2:4]
        push!(positive_images, (matrix[i][1], recortar(recorte, matrix[i][2])))
    end
end

#display(positive_images[1])

# Recorrer negative_dataset
    # Para cada negative_dataset[i], aplicarle a images[i] los recortes y apilarlos en positive_images[] de la siguiente manera push!(positive_images, recorte)

# matrix_c = matrix[1]

# coord = [convert(UInt16,negative_dataset[1][2]), convert(UInt16,negative_dataset[1][3]), convert(UInt16,round(negative_dataset[1][4])) ] #x, y, radius

# matrix_cut = matrix_c[coord[1],coord[2]]

# println(matrix_cut)

# Crear el directorio si no existe
if !isdir(hit_path)
    mkdir(hit_path)
end
global cont = 1
for (name2, image) in positive_images
    # Guardar la imagen en el directorio
    name = "recorte"*string(cont)*".png"
    println("saving \""*name*"\" of image named :"*name2)
    save(joinpath(hit_path, name), image)
    global cont = cont+1
end

# Crear el directorio si no existe
if !isdir(miss_path)
    mkdir(miss_path)
end
global cont = 1
for (name2, image) in negative_images
    # Guardar la imagen en el directorio
    name = "recorte"*string(cont)*".png"
    println("saving \""*name*"\" of image named :"*name2)
    save(joinpath(miss_path, name), image)
    global cont = cont+1
end


#-----------------------------------Extraccion de caracteristicas----------------------------------------------------------

