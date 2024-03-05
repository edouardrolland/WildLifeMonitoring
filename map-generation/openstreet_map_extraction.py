import geojson
import matplotlib.pyplot as plt
import shapely
import trajgenpy.Logging
from trajgenpy import Utils
from trajgenpy.Geometries import GeoMultiPolygon, GeoMultiTrajectory, GeoPolygon
from trajgenpy.Query import query_features
import osmnx as ox


log = trajgenpy.Logging.get_logger()

if __name__ == "__main__":
    # Download the water features data within the bounding box
        
    # Spécifiez la relation ID
    relation_id = 13764596

    # Récupérer la géométrie à partir de la relation ID
    
    features = ox.features_from_place("Ol Pejeta Conservancy", which_result=None, tags={"name": ["Ol Pejeta Conservancy"]}, buffer_dist=None)

    # Amagerværket: shapely.Polygon([(12.620400,55.687962),(12.632788,55.691589),(12.637446,55.687689),(12.624924,55.683489)])
    # Davinde: shapely.Polygon([(10.490913, 55.315346), (10.576744, 55.315346), (10.576744, 55.337417), (10.490913, 55.337417)])

    
    filtered_features = features[features["name"].notna()]
    polygon = GeoPolygon(
        filtered_features.geometry.unary_union,
        crs="WGS84",
    )  # Download the water features data within the bounding box
    print(polygon)
    polygon.plot(facecolor="none", edgecolor="black", linewidth=2)
    
    
    
    # Plot natural features
    # coastline = GeoMultiTrajectory(features["natural"], crs="WGS84").set_crs(
    #     "EPSG:2197"
    # )
    # coastline.plot(color="green")

    # Export the polygon and the buildings as a GeoJSON file
    geojson_collection = geojson.FeatureCollection(
        [
            polygon.to_geojson(id="boundary"),
        ]
    )

    with open("environment.geojson", "w") as f:
        geojson.dump(geojson_collection, f)

    # Plot on a map
    Utils.plot_basemap(crs="EPSG:2197")

    # No axis on the plot
    plt.axis("equal")
    plt.axis("off")
    plt.show()