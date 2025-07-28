def get_planet_name(img_path):
    file_name = img_path.split('/')[-1]
    planet = file_name.split('_')[0]
    return planet

def evaluate_latent_space(analyzer_obj):
    n_clusters = analyzer_obj.n_clusters
    planets_to_cluster = {}
    for img_path in analyzer_obj.image_paths:
        planet = get_planet_name(img_path)
        planets_to_cluster[planet] = [0 for i in range(n_clusters)] # no of samples in each cluster

    for i in range(len(analyzer_obj.image_paths)):
        planet = get_planet_name(analyzer_obj.image_paths[i])
        cl = analyzer_obj.cluster_labels[i]
        planets_to_cluster[planet][cl] += 1
    
    tot_err = 0
    cnt = 0
    history = {}
    for planet in list(planets_to_cluster.keys()):
        a = planets_to_cluster[planet]
        mx = max(a)
        s = sum(a)
        rem = s-mx
        err = rem/s
        if (s==1):
            continue
        tot_err += err
        cnt += 1
        history[planet] = err
    avg_err = tot_err/cnt
    return avg_err, history