include("funciones.jl")

#---------------------------------------- Carga de recortes -----------------------------------------------------------

# Carga de Hit 'n Miss

hit = loadFolderImages(hit_path)
#hit = positive_images

miss = loadFolderImages(miss_path)
#miss = negative_images

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
