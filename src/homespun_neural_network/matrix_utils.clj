(ns homespun-neural-network.matrix-utils
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as M]))

(defn sum-rows
  [mat]
  (map (partial apply +)
       (m/rows mat)))

(defn vector-mean
  [v]
  (let [m (count v)]
    (->> v
         (apply +)
         (* (/ 1 m)))))

(defn matrix-row-mean
  [mat]
  (let [m (-> mat m/shape second)]
    (->> mat
         m/rows
         (map (partial apply +))
         (M/* (/ 1 m)))))
