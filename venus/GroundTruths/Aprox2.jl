include("funciones.jl")

#---------------------------------------- Carga de recortes -----------------------------------------------------------

# Carga de Hit 'n Miss

hit = loadFolderImages(hit_path)
#hit = positive_images

miss = loadFolderImages(miss_path)
#miss = negative_images

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

for image in hit1
    push!(third_part, featureExtraction(image, 0, [0, 1]))
end

for image in miss1
    push!(fourth_part, featureExtraction(image, 1, [0, 1]))
end

dataSet1 = vcat(first_part, second_part)
dataSet2 = vcat(third_part, fourth_part)

aux = hcat(dataSet1, dataSet2)

dataSet = []

for fila in aux
    i = 0
    for element in fila
    push!(dataSet[i], element)
    end
    i = i+1
end

# println(typeof(dataSet)) -> Matrix{Vector{Float64}}

#normalizeMinMax!(dataSet[1])

# Guardamos el dataSet en "aprox1.data"

saveAsData(path*"/"*"aprox2.data", dataSet, '\t')


function calcularHist(img::Array{Float64, 2})
    h = zeros(1,256);
    println("Calculating Histogram on image with size "*string(size(img, 1))*"x"*string(size(img, 2)))
    for x in img
        i = round(1 + x*255, digits = 0)
        i= Int(i)
        h[i] = h[i]+1
    end
    return h
end