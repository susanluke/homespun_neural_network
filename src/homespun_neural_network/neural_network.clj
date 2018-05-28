(ns homespun-neural-network.neural-network
  (:require [homespun-neural-network.matrix-utils :as hnn-m]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as M]
            [clojure.core.matrix.random :as mr]))

;; TODO - not sure it's nice to store grads & net params in
;; maps.  Maybe change to storing in vectors and using numerical
;; indexes.

;; Hyperparameters
(def initial-weight-scale 0.0001)

(defn num-layers [m]
  (count (:layer-sizes m)))

(defn relu [m] (m/emap #(max 0 %) m))
(defn relu-grad [m] (m/emap #(if (< 0 %) 1 0) m))
(defn logistic-grad [m] (m/emap #(* % (- 1 %)) m))

(def activation-fns {:relu {:fn relu
                            :dfn relu-grad}
                     :sigmoid {:fn m/logistic
                               :dfn logistic-grad}
                     :identity {:fn identity ; only for test purposes
                                :dfn (constantly 1)}})

(defn make-key [k n] (keyword (str (name k) n)))

(defn init-net-params
  "Takes:
  layer-sizes - list containing number of nodes in each layer"
  [layer-sizes]
  (let [num-layers (count layer-sizes)]
    (loop [net-params {:layer-sizes layer-sizes
                       :W [nil]
                       :b [nil]}
           idx 1]
      (if (= num-layers idx)
        net-params
        (let [random-numbers (mr/randoms idx)
              W-rows (nth layer-sizes idx)
              W-cols (nth layer-sizes (dec idx))
              W (-> (take (* W-rows W-cols) random-numbers)
                    (m/reshape [W-rows W-cols])
                    (M/* initial-weight-scale))
              b 0] ; set b to 0, rely on broadcasting to scale up
          (recur (assoc net-params
                        :W (conj (:W net-params) W)
                        :b (conj (:b net-params) b))
                 (inc idx)))))))

(defn update-net-params
  [net-params grads learning-rate]
  (let [num-layers (num-layers net-params)]
    (loop [new-net-params {}
           idx 1]
      (if (= num-layers idx)
        new-net-params
        (let [W-key (keyword (str "W" idx))
              W-old (W-key net-params)
              dW ((keyword (str "dW" idx)) grads)
              W (M/- W-old (M/* learning-rate dW))
              b-key (keyword (str "b" idx))
              b-old (b-key net-params)
              db ((keyword (str "db" idx)) grads)
              b (M/- b-old (M/* learning-rate db))]
          (recur (assoc new-net-params
                        W-key W
                        b-key b)
                 (inc idx)))))))

(defn forward-prop
  [X net-params]
  (let [num-layers (num-layers net-params)]
    (loop [state {:A0 X}
           idx 1]
      (println "[forward-prop] idx:" idx)
      (if (= num-layers idx)
        state
        (recur
         (let [W-key (make-key :W idx)
               b-key (make-key :b idx)
               A-key (make-key :A idx)
               Z-key (make-key :Z idx)
               fn-key (make-key :fn idx)
               A-prev-key (make-key :A (dec idx))
               W (W-key net-params)
               b (b-key net-params)
               activation-fn (fn-key net-params)
               Z (->> (A-prev-key state)
                      (m/dot W)
                      (M/+ b))
               A (activation-fn Z)]
           (assoc state Z-key Z A-key A))
         (inc idx))))))

(defn cost
  "Y & A are matrices of shape [1 m]
  Returns the logisitic cost based on formula:
  L = (-1*m).sum[y.log(a) + (1-y).log(1-a)]"
  [Y A]
  (let [m (-> Y m/shape second)] ; number of columns in Y
    (* (/ -1 m)
       (m/esum (M/+
                (M/* Y (m/log A))
                (M/* (M/- 1 Y) (m/log (M/- 1 A))))))))


(defn back-prop
  "Given current activation state for the network, calculates and
  returns gradients for each weight and bias"
  [X Y state net-params]
  (let [layers (num-layers net-params)
        m (-> Y m/shape second)
        A ((make-key :A (dec layers)) state)
        dZl (m/esum (M/- A Y))]
    (loop [l (dec layers)
           grads {(make-key :dZ l) dZl}]
      (if (= 0 l)
        grads
        (recur
         (dec l)
         (let [dZ     ((make-key :dZ l) grads)
               A-prev ((make-key :dA (dec l)) state)
               W      ((make-key :W l) net-params)
               Z-prev ((make-key :Z (dec l)) state)
               dfn-prev (:dfn ((make-key :fn (dec l)) net-params))
               dW (-> dZ
                      (m/dot (m/transpose A-prev))
                      hnn-m/matrix-row-mean)
               db (hnn-m/matrix-row-mean dW)
               dA-prev (->> dZ
                            (m/dot (m/transpose A))
                            hnn-m/matrix-row-mean)
               dZ-prev (->> dA-prev
                            (M/* (m/emap dfn-prev Z-prev)))]
           (assoc grads
                  (make-key :dW l) dW
                  (make-key :db l) db
                  (make-key :dA (dec l)) dA-prev
                  (make-key :dZ (dec l)) dZ-prev)))))))

(defn neural-network
  [X Y net-params learning-rate num-iterations]
  (if (= 0 num-iterations)
    net-params
    (let [state (forward-prop X net-params)
          A ((make-key :A (num-layers net-params)) state)
          cost (cost Y A)
          grads (back-prop X Y state net-params)
          new-net-params (update-net-params net-params grads learning-rate)]
      (recur X Y net-params learning-rate num-iterations))))
