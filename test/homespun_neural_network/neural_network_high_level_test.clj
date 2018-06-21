(ns homespun-neural-network.neural-network-high-level-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as M]
            [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as sgen]
            [clojure.test.check.generators :as tg]
            [homespun-neural-network.neural-network :as sut]))

;; TODO - no actual testing here..
(deftest train-neural-network-test
  (comment (time (let [net-params {:layer-sizes [2 3 1],
                                   :fns [:identity :relu :sigmoid],
                                   :W
                                   [nil
                                    [[7.308781907032909E-5 4.100808114922017E-5]
                                     [2.077148413097171E-5 3.327170559595112E-5]
                                     [9.677559094241208E-5 6.1171822657613E-7]]
                                    [[7.311469360199059E-5
                                      9.014476240300544E-5
                                      4.9682259343089074E-5]]],
                                   :b [nil [[0.0] [0.0] [0.0]] [[0.0]]]}
                       X [[1 3] [10 20]]
                       Y [[0 1]]
                       learning-rate 1e-2
                       num-iterations 5000
                       new-net-params (sut/train-neural-network X Y
                                                                net-params
                                                                learning-rate
                                                                num-iterations)]
                   (println "new-net-params" new-net-params)))))

(defn my-simple-function1 [x1 x2]
  (if (>= (+ (* 0.8 x1) 0.05) x2)
    1 0))


(defn raw-fn [x1 x2]
  (+ (* 0.55 x1) (* -0.455 x2) -0.01))

(defn my-simple-function [x1 x2]
  (if (>= (raw-fn x1 x2) 0)
    1 0))

(defn input->y
  [f input]
  [input (apply f input)])


(def num-gen (s/gen float?))


(s/def ::num-0-1 (s/with-gen (partial s/int-in-range? 0 1)
                   #(s/gen float?)))
(s/def ::perceptron-params (s/tuple ::num-0-1 ::num-0-1))

(def data-gen (sgen/fmap (partial input->y my-simple-function)
                         (s/gen ::perceptron-params)))

(defn data->x-y
  [data]
  (let [m (count data)
        x (m/transpose (map first data))
        y (m/reshape  (map second data) [1 m])]
    [x y]))

(deftest train-perceptron-test
  (time (let [net-params (sut/init-net-params [2 1] [:identity :sigmoid])
              net-params {:layer-sizes [2 1],
                          :fns [:identity :sigmoid],
                          :W [nil [[0.55 -0.455]]],
                          :b [nil [[-0.01]]]}
              _ (println "initial net params" net-params)
              data (data->x-y (sgen/sample data-gen 256))
              X (first data)
              Y (second data)
              _ (println "X:" (m/shape X))
              _ (println "Y:" (m/shape Y))
              learning-rate 1e-2
              num-iterations 3000
              new-net-params (sut/train-neural-network X Y
                                                       net-params
                                                       learning-rate
                                                       num-iterations)]
          (println "new-net-params" new-net-params)
          (is true))))

(comment
  "
  0.8x1 + 0.05 >= x2

  0.8x1 + 0.3x2 -0.2 > 0.5

"
  (data->x-y (sgen/sample data-gen 10)))
