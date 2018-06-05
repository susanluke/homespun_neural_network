(ns homespun-neural-network.matrix-utils
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as M]))

;; TODO: requires tests

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

(defn matrix-row-mean-keep-dims
  [mat]
  (let [m (-> mat m/shape second)]
    (map (comp
          vector
          (partial * (/ 1 m))
          (partial apply +))
         (m/rows mat))))

(defn broadcast-sideways
  [mat n-cols]
  (apply (partial m/join-along 1) ; broadcast b out to match Z dims
         (repeat n-cols mat)))
