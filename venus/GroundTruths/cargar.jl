# Import packages

import Pkg;

using DelimitedFiles

#Pkg.add("JLD2")
using JLD2

#Pkg.add("Plots")
using Plots

#Pkg.add("Colors")
using Colors

#Pkg.add("FileIO")
using FileIO

#Pkg.add("Images")
using Images

#Pkg.add("NaturalSort")
using NaturalSort

#Pkg.add("ImageView")
#using ImageView

#Pkg.add("Statistics")
using Statistics

#Pkg.add("ScikitLearn")
using ScikitLearn

#Pkg.add("Flux")
using Flux
using Flux.Losses

#Pkg.add("Random")
using Random
using Random:seed!

include("funciones.jl")

# PATHS

pattern_path = mycd*"Venus_AA/venus/GroundTruths/Patrones"
img_path = mycd*"Venus_AA/venus/imagenes"
hit_path = mycd*"Venus_AA/venus/hit"
miss_path = mycd*"Venus_AA/venus/miss"
path = mycd*"Venus_AA/venus"


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
    for j=1:(size(negative_dataset[i], 1))
        println(negative_dataset[i][j])
        recorte = negative_dataset[i][j][2:4]
        push!(negative_images, recortar(recorte, matrix[i]))
    end
end

for i=1:size_data
    for j=1:(size(positive_dataset[i], 1))
        println(positive_dataset[i][j])
        recorte = positive_dataset[i][j][2:4]
        push!(positive_images, recortar(recorte, matrix[i]))
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
for (image) in positive_images
    # Guardar la imagen en el directorio
    name = "recorte"*string(cont)*".png"
    println("saving \""*name*"\"")
    save(joinpath(hit_path, name), image)
    global cont = cont+1
end

# Crear el directorio si no existe
if !isdir(miss_path)
    mkdir(miss_path)
end
global cont = 1
for (image) in negative_images
    # Guardar la imagen en el directorio
    name = "recorte"*string(cont)*".png"
    println("saving \""*name*"\"")
    save(joinpath(miss_path, name), image)
    global cont = cont+1
end


#---------------------------------------- Carga de recortes -----------------------------------------------------------

# Carga de Hit 'n Miss

#hit = loadFolderImages(hit_path)
hit = positive_images

#miss = loadFolderImages(miss_path)
miss = negative_images

#---------------------------------- Extraccion de caracteristicas -----------------------------------------------------

first_part = []

for image in hit
    push!(first_part, featureExtraction(image, 0, [0, 1]))
end

second_part = []
for image in miss
    push!(second_part, featureExtraction(image, 1, [0, 1]))
end

dataSet = vcat(first_part, second_part)

#normalizeMinMax!(dataSet[1])

# Guardamos el dataSet en "aprox1.data"

saveAsData(path*"/"*"aprox1.data", dataSet, '\t')

# TODO Mirar una alternativa a los circulos azules para los plots de latex,
# TODO algo como +

#------------------------------------- Carga del ".data" --------------------------------------------------------------


dataSetImportado = readdlm(path*"/"*"aprox1.data", '\t')



#----------------------------------- Creacion de los modelos ----------------------------------------------------------

# Importación de los modelos que vamos a emplear

# Se hace en "funciones.jl"

# @sk_import svm: SVC
# @sk_import tree: DecisionTreeClassifier
# @sk_import neighbors: KNeighborsClassifier 



#------------------------------------- Código final -------------------------------------------------------------------


salida = open("resultados_aprox_1.txt", "w") #Esto lo vamos a ir cambiando en cada aproximación

numFolds = 10 

topology=[3,4]
learningRate = 0.01
maxEpochs = 100 #1000
validationRatio = 0.2   # Si queremos que el modelo no se entrene usando un conjunto de validación, este valor debe ser 0
maxEpochsVal = 6
numRepetitions = 5 #50

kernel = "rbf"
kernelDegree = 3
kernelGamma = 2
C=1

maxDepth = 4

numNeighbors = 8

#Dividimos el dataset en dos (features y targets)
inputs = convert(Array{Float64, 2}, dataSetImportado[:, 1:2]) 
targets = reshape((dataSetImportado[:, 3]), :, 1)

#PDF's
# inputs = dataSetImportado[:, 1:2];
# inputs = convert(Array{Float32, 2}, inputs);
# targets = dataSetImportado[:, 3];
# targets = convert(Array{Bool, 1}, targets);

normalizeMinMax!(inputs);

# Dicionario con los hiperparametros utilizados para el entrenamiento de RNA

modelHyperparameters = Dict();
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitions;
modelHyperparameters["maxEpochs"] = maxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;


# RNA

topologies = [[3], [1], [2], [1,1], [1,2], [3,4], [2,3], [5,5]]

for i in topologies
    modelHyperparameters["topology"] = i
    macc, sacc, mf1, sf1 = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, numFolds);
    write(salida, string("\nRNA: ", string(i), "\nMedia de precision del test en 10-fold: ", round(macc*100, digits=2)," \nDesviacion tipica del test: ", round(sacc*100, digits=2), "\nMedia de Especificidad de test ", round(mf1*100, digits=2), "\nDesviacion tipica de Especificidad de test ", round(sf1*100, digits=2)))
end

# Hiperparametros utilizados para el entrenamiento de SVM

modelHyperparameters = Dict();
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;


# SVM

topologies = [("rbf", 0.1), ("rbf", 0.6), ("rbf", 1), ("poly", 0.1), ("poly", 0.6), ("poly", 1), ("sigmoid", 0.1), ("sigmoid", 0.6), ("sigmoid", 1)]

for topology in topologies
    modelHyperparameters["kernel"] = topology[1];
    modelHyperparameters["C"] = topology[2];
    macc, sacc, mr1, sr1, mf1, sf1 = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
    write(salida, string("\nSVM: con kernel \"", string(topology[1]), "\" y C:", string(topology[2]), "\nMedia de precision del test en 10-fold: ", round(macc*100, digits=2)," \nDesviacion tipica del test: ", round(sacc*100, digits=2), 
    "\nMedia de Sensibilidad de test ", round(mr1*100, digits=2), "\nDesviacion tipica de Sensibilidad de test ", round(sr1*100, digits=2),
    "\nMedia de Especificidad de test ", round(mf1*100, digits=2), "\nDesviacion tipica de Especificidad de test ", round(sf1*100, digits=2)))
end


# Arbol de Decision

depths = [1, 3, 4, 5, 6, 7, 10]

for depth in depths
    macc, sacc, mr1, sr1, mf1, sf1 = modelCrossValidation(:DecisionTree, Dict("maxDepth" => depth), inputs, targets, numFolds);
    write(salida, string("\nDECISION TREE: ", string(depth), "\nMedia de precision del test en 10-fold: ", round(macc*100, digits=2)," \nDesviacion tipica del test: ", round(sacc*100, digits=2),
    "\nMedia de Sensibilidad de test ", round(mr1*100, digits=2), "\nDesviacion tipica de Sensibilidad de test ", round(sr1*100, digits=2),
    "\nMedia de Especificidad de test ", round(mf1*100, digits=2), "\nDesviacion tipica de Especificidad de test ", round(sf1*100, digits=2)))
end


# KNN

neighboors = [2,5, 8, 11, 14, 17]

for i in neighboors
    macc, sacc, mr1, sr1, mf1, sf1= modelCrossValidation(:kNN, Dict("numNeighbors" => i), inputs, targets, numFolds);
     write(salida, string("\nKNN: ", string(i), "\nMedia de precision del test en 10-fold: ",macc," \nDesviacion tipica del test: ", sacc, 
     "\nMedia de Sensibilidad de test ", mr1, "\nDesviacion tipica de Sensibilidad de test ", sr1,
     "\nMedia de Especificidad de test ", mf1, "\nDesviacion tipica de Especificidad de test ", sf1))
end


close(salida)