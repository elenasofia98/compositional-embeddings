class ClusterMinDiam:
    @staticmethod
    def _farest(values, centers):
        max = 0
        for value in values:
            d_value = min([abs(value - c) for c in centers])
            if d_value > max:
                max = d_value
                next = value

        return next

    @staticmethod
    #2 approx
    def _k_clusters_of_min_diameter(k, values):
        centers = [values[0]]
        for i in range(1, k):
            next = ClusterMinDiam._farest(values, centers)
            centers.append(next)

        clusters = {}
        for center in centers:
            clusters[center] = []

        for value in values:
            min = None
            for center in centers:
                d_center_value = abs(value - center)
                if min is None or d_center_value < min:
                    min = d_center_value
                    chosen_centre = center
            clusters[chosen_centre].append(value)

        return clusters

    @staticmethod
    def k_clusters_of_min_diameter(k, n_clusters: dict):
        k_clusters = {}
        if len(n_clusters.keys()) <= k:
            for center in n_clusters:
                k_clusters[center] = [(center, x) for x in n_clusters[center]]
            return k_clusters

        clusterized_values = ClusterMinDiam._k_clusters_of_min_diameter(k, [x for x in n_clusters])
        for center in clusterized_values:
            k_clusters[center] = []
            for value in clusterized_values[center]:
                k_clusters[center].extend([(value, x) for x in n_clusters[value]])
        return k_clusters