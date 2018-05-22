(ns homespun-neural-network.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as M])
  (:gen-class))

(defn sigmoid
  [x]
  (m/logistic x))

(defn sum-rows
  [mat]
  (map (partial apply +)
       (m/rows mat)))

(defn vector-mean
  [vec]
  (let [m (count vec)]
    (->> vec
         (apply +)
         (* (/ 1 m)))))

(defn matrix-row-mean
  [mat]
  (let [m (-> mat m/shape second)]
    (->> mat
         m/rows
         (map (partial apply +))
         (m/emul (/ 1 m)))))

(defn init-params
  "Returns initial values for W and b, given the number of dimensions
  in x"
  [n-x]
  {:W (m/zero-matrix n-x 1)
   :b 0})

(defn forward-prop
  "Returns A for given X, W & b"
  [X W b]
  (->> X
       (m/dot (m/transpose W))
       (m/add b)
       sigmoid))

(defn cost
  [Y A]
  (let [m (count Y)]
    (* (/ -1 m) (m/esum (m/add (m/emul Y
                                       (m/log A))
                               (m/emul (m/sub 1 Y)
                                       (m/log (m/sub 1 A))))))))

(defn back-prop
  "Given A, Y & X, returns gradients for W and b"
  [A Y X]
  (let [m (count Y)
        dZ (m/sub A Y)
        dW (matrix-row-mean (m/emul dZ X))
        db (vector-mean dZ)]
    {:dW dW
     :db db}))

(defn update-params
  "Given param"

  [{:keys [W b] :as params}
   {:keys [dW db] :as grads}
   learning-rate]
  {:W (- W (* learning-rate dW))
   :b (- b (* learning-rate db))})


(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
