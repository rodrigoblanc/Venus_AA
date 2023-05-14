# Import packages

import Pkg;

using DelimitedFiles

#Pkg.add("JLD2")
#using JLD2

#Pkg.add("Plots")
#using Plots

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

# Incluimos las funciones para poder ejecutar las "operaciones" posteriores
include("funciones.jl")


############################ Preprocesado de datos #############################

include("Preprocesado.jl")


########################### Selección de aproximación ##########################

#AQUI SE ESCRIBE LA APROXIMACION QUE SE QUIERE EJECUTAR
nAprox = "2"
#miss_path = mycd*"Venus_AA/venus/cuts"

############################# Recortes de patrones #############################

#=
    Para la aproximación N hacen falta ejecutar los recortes anteriores, así que
    por simplicidad, es tarea del usuario, determinar que recortes hace falta
    obtener y cuales no
=#

if nAprox == "1"
    recortars = ["1"]

elseif nAprox == "2"
    recortars = ["1", "2"]
elseif nAprox == "3"
    recortars = ["1", "2", "3"]
else
    recortars = ["1", "2", "3", "4"]
end

for i in recortars
    include("Recortar"*string(i)*".jl")
end

########################### Ejecucción de aproximación #########################

include("Aprox"*nAprox*".jl")


########################### Generación del ".data" #############################

#include("Aprox"*nAprox*".jl")

# Debe ser común a todos los PC's
#ccd("Venus_AA/venus/GroundTruths/dataset_etiquetado")


########################### Carga del ".data" ##################################


#dataSetImportado = readdlm(path*"/"*"aprox"*nAprox*".data", '\t')
#println(dataSetImportado)


######################### Creacion de los modelos ##############################

# Importación de los modelos que vamos a emplear

# Se hace en "funciones.jl"

# @sk_import svm: SVC
# @sk_import tree: DecisionTreeClassifier
# @sk_import neighbors: KNeighborsClassifier 
