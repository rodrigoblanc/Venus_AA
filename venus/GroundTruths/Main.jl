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

# include("Recortar.jl")

nAprox = "2"
# include("Aprox"*nAprox*".jl")

# Debe ser común a todos los PC's
#ccd("Venus_AA/venus/GroundTruths/dataset_etiquetado")


#------------------------------------- Carga del ".data" --------------------------------------------------------------


dataSetImportado = readdlm(path*"/"*"aprox"*nAprox*".data", '\t')
println(dataSetImportado)

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
    macc, sacc, mf1, sf1 = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
    write(salida, string("\nSVM: con kernel \"", string(topology[1]), "\" y C:", string(topology[2]), "\nMedia de precision del test en 10-fold: ", round(macc*100, digits=2)," \nDesviacion tipica del test: ", round(sacc*100, digits=2), "\nMedia de Especificidad de test ", round(mf1*100, digits=2), "\nDesviacion tipica de Especificidad de test ", round(sf1*100, digits=2)))
end


# Arbol de Decision

depths = [1, 3, 4, 5, 6, 7, 10]

for depth in depths
    macc, sacc, mf1, sf1 = modelCrossValidation(:DecisionTree, Dict("maxDepth" => depth), inputs, targets, numFolds);
    write(salida, string("\nDECISION TREE: ", string(depth), "\nMedia de precision del test en 10-fold: ", round(macc*100, digits=2)," \nDesviacion tipica del test: ", round(sacc*100, digits=2), "\nMedia de Especificidad de test ", round(mf1*100, digits=2), "\nDesviacion tipica de Especificidad de test ", round(sf1*100, digits=2)))
end


# KNN

neighboors = [2,5, 8, 11, 14, 17]

for i in neighboors
    macc, sacc, mf1, sf1= modelCrossValidation(:kNN, Dict("numNeighbors" => i), inputs, targets, numFolds);
     write(salida, string("\nKNN: ", string(i), "\nMedia de precision del test en 10-fold: ",macc," \nDesviacion tipica del test: ", sacc, "\nMedia de Especificidad de test ", mf1, "\nDesviacion tipica de Especificidad de test ", sf1))
end


close(salida)