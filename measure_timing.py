import platform
import statistics
import time

from poc_recommender import _catalog_and_vectors, recommend

_catalog_and_vectors.cache_clear()
start = time.perf_counter()
recommend("The Matrix")
first = time.perf_counter() - start

repeats = []
for _ in range(20):
    start = time.perf_counter()
    recommend("The Matrix")
    repeats.append(time.perf_counter() - start)

movies, _ = _catalog_and_vectors()
print(f"Machine: {platform.platform()}, Python {platform.python_version()}")
print(f"Catalog: {len(movies)} movies")
print(f"First request (loads + fits): {first:.3f} s")
print(f"Median repeated request:      {statistics.median(repeats):.4f} s")