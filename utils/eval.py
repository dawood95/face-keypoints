import cv2
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def can_merge(person1, person2):
    for k in person1:
        if person1[k] != person2[k] and not person1[k] is None and not person2[k] is None:
            return False
    return True

def get_line(p1, p2):
    "Bresenham's line algorithm"
    x0, y0 = p1
    x1, y1 = p2
    pts = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            pts.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            pts.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    pts.append((x, y))
    return pts

def hm2kpts(heatmap, jointmap, img_id, hm_threshold=0.1, jm_threshold=0.05, scale=(8, 8), sigma=1):

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    kpts = [[] for _ in range(17)]
    for part in range(17):
        map_ori = heatmap[part, :,:]
        map     = gaussian_filter(map_ori, sigma=1)

        map_left         = np.zeros(map.shape)
        map_left[1:,:]   = map[:-1,:]
        map_right        = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up           = np.zeros(map.shape)
        map_up[:,1:]     = map[:,:-1]
        map_down         = np.zeros(map.shape)
        map_down[:,:-1]  = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > hm_threshold))
        peaks        = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
        kpts[part]   = peaks

    people = []
    for idx, (i, j) in enumerate(skeleton):
        pts1 = kpts[i - 1]
        pts2 = kpts[j - 1]
        jm = (jointmap[idx] ** 2 + jointmap[idx + 19] ** 2) ** 0.5
        #jm = cv2.dilate(cv2.resize(jointmap[idx], (iw, ih)), kernel)
        pairs = {}
        for m, pt1 in enumerate(pts1):
            for n, pt2 in enumerate(pts2):
                pts = np.array(get_line(pt1, pt2))
                score = jm[pts[:, 1], pts[:, 0]].mean()
                if score < jm_threshold: continue
                pairs[(m, n)]= score
        pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)

        unmatched_pairs = []
        for (m, n), score in pairs:
            found = None
            person = None
            idx = None
            for _idx, _person in enumerate(people):
                p_m = _person[i - 1]
                p_n = _person[j - 1]
                if found == 'M':
                    if p_n == n:
                        if not p_m is None:
                            found = 'P'
                            break
                        elif not can_merge(_person, person):
                            found = 'P'
                            break
                    continue
                elif found == 'N':
                    if p_m == m:
                        if not p_n is None:
                            found = 'P'
                            break
                        elif not can_merge(_person, person):
                            found = 'P'
                            break
                    continue
                if p_m == m and p_n == None:
                    found = 'M'
                    person = _person
                    idx = _idx
                    continue
                elif p_n == n and p_m == None:
                    found = 'N'
                    person = _person
                    idx = _idx
                    continue
                elif p_m == m or p_n == n:
                    found = 'P'
                    break

            if found is None and score > (jm_threshold * 2):
                person = {}
                for k in range(17): person[k] = None
                person[i - 1] = m
                person[j - 1] = n
                people.append(person)
            elif found is 'M':
                person[j - 1] = n
                for p in people:
                    if p == person: continue
                    if p[j - 1] == n:
                        for k in range(17):
                            if p[k] is None:
                                p[k] = person[k]
                            elif not person[k] is None and not person[k] == p[k]:
                                raise
                        people.pop(idx)
                        break
            elif found is 'N':
                person[i - 1] = m
                for p in people:
                    if p == person: continue
                    if p[i - 1] == m:
                        for k in range(17):
                            if p[k] is None:
                                p[k] = person[k]
                            elif not person[k] is None and not person[k] == p[k]:
                                raise
                        people.pop(idx)
                        break

    r = []
    for p in people:
        person = []
        for k in p:
            if p[k] is None: person.extend([0, 0, 0]); continue
            x, y = kpts[k][p[k]]
            person.extend([int(x * scale[0]), int(y * scale[1]), 2])
        r.append({
            'image_id'    : img_id,
            'category_id' : 1,
            'keypoints'   : person,
            'score'       : 1
        })

    return r
