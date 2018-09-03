(ns homespun-neural-network.neural-network
  (:require [homespun-neural-network.matrix-utils :as hnn-m]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as M]
            [clojure.core.matrix.random :as mr]))

;; Hyperparameters
(def initial-weight-scale 0.0001)

(defn num-layers [m]
  (count (:layer-sizes m)))

;;; Activation functions and gradient functions
(defn relu [m] (m/emap #(max 0 %) m))
(defn relu-grad [m] (m/emap #(if (< 0 %) 1 0) m))
(defn identity-in-range [m] (m/emap #(max 0 (min 1 %)) m))
(defn identity-in-range-grad [m] (m/emap #(if (< 0 % 1) % 0)))
(defn logistic-grad
  [m]
  (m/emap #(* (m/logistic %)
              (- 1 (m/logistic %)))
          m))

(def activation-fns {:relu {:fn relu
                            :dfn relu-grad}
                     :sigmoid {:fn m/logistic
                               :dfn logistic-grad}
                     :identity {:fn identity ; only for test purposes
                                :dfn (constantly 1)}
                     :identity-in-range {:fn identity-in-range
                                         :dfn identity-in-range-grad}})

(defn init-net-params
  "Takes:
  layer-sizes - list containing number of nodes in each layer
                (including layer 0)
  fns - list containing activation fn for each layer
  Returns:
  net-params - a map containing
      :layer-sizes - vector containing number of nodes in each layer
      :W - a vector containing nil, W1, W2, ... for each layer
      :b - a vector containing nil, b1, b2, ... for each layer
  NB initial value in :W, :b is nil to make indexing by layer simpler. "
  [layer-sizes fns]
  (let [num-layers (count layer-sizes)]
    (loop [net-params {:layer-sizes layer-sizes
                       :fns fns
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
              b (m/zero-matrix W-rows 1)]
          (recur (assoc net-params
                        :W (conj (:W net-params) W)
                        :b (conj (:b net-params) b))
                 (inc idx)))))))

;; TODO: Maybe we can just map over the arrays, rather than loop/recur?
(defn update-net-params
  [{:keys [layer-sizes fns] :as net-params} grads learning-rate]
  (let [num-layers (num-layers net-params)]
    (loop [new-net-params {:layer-sizes layer-sizes
                           :fns fns
                           :W [nil]
                           :b [nil]}
           idx 1]
      (if (= num-layers idx)
        new-net-params
        (let [W (nth (:W net-params) idx)
              b (nth (:b net-params) idx)
              dW (nth (:dW grads) idx)
              db (nth (:db grads) idx)
              W-new (M/- W (M/* learning-rate dW))
              b-new (M/- b (M/* learning-rate db))]
          (recur (assoc new-net-params
                        :W (conj (:W new-net-params) W-new)
                        :b (conj (:b new-net-params) b-new))
                 (inc idx)))))))

;; TODO: poss re-write as reduce?
(defn forward-prop
  [X net-params]
  (let [num-layers (num-layers net-params)
        m (-> X m/shape second)]
    (loop [state {:A [X]
                  :Z [nil]}
           idx 1]
      (if (= num-layers idx)
        state
        (let [W      (nth (:W net-params) idx)
              b      (nth (:b net-params) idx)
              A-prev (nth (:A state) (dec idx))
              fn-key (nth (:fns net-params) idx)
              a-fn   (get-in activation-fns [fn-key :fn])
              Z      (->> A-prev
                          (m/dot W)
                          (M/+ (apply (partial m/join-along 1) ; broadcast b out to match Z dims
                                      (repeat m b))))
              A      (a-fn Z)]
          (recur (assoc state
                        :Z (conj (:Z state) Z)
                        :A (conj (:A state) A))
                 (inc idx)))))))

(defn cost
  "Y & A are matrices of shape [1 m]
  Returns the logistic cost based on formula:
  L = (-1*m).sum[y.log(a) + (1-y).log(1-a)]"
  [Y A]
  (let [m (-> Y m/shape second)] ; number of columns in Y
    (* (/ -1 m)
       (m/esum (M/+
                (M/* Y (m/log A))
                (M/* (M/- 1 Y) (m/log (M/- 1 A))))))))

(defn accuracy
  [Y A]
  (let [A*  (m/emap #(if (> % 0.5) 1 0) A)
        res (m/emap = Y A*)
        m   (second (m/shape Y))
        num-correct (count (filter true? (first res)))]
    (float (/ (* num-correct 100) m))))

;; TODO: Not using this function?  Get rid? Or put initial dJ/dA val in
;; backprop?
(defn cost-grad
  "Returns d(cost(Y,A)/dA
  returns a matrix of [[1]]"
  [Y A]
  (hnn-m/matrix-row-mean-keep-dims (M/- (M// Y A)
                                        (M// (M/- 1 Y)
                                             (M/- 1 A)))))

;; TODO - maybe just calc dA prior to loop/recur
;; then need to calc dZ dW dB and dA-prev
(defn back-prop
  "Given current activation state for the network, calculates and
  returns gradients for each weight and bias"
  [X Y net-params state]
  (let [layers (num-layers net-params)
        m (-> Y m/shape second)
        A-l (last (:A state))
        dA-l (cost-grad Y A-l)
        dZ-l (M/- A-l Y)]
    (loop [l (dec layers)
           ;; use lists so conj adds to the head
           grads {:dZ (list dZ-l)
                  :dA '(nil)  ; don't bother calculating dJ/dA
                  :dW '()
                  :db '()}]
      (if (= 0 l)
        (assoc grads
               ;; Add in dW-0 and db-0 so indexes match up
               :dW (conj (:dW grads) nil)
               :db (conj (:db grads) nil))
        (recur
         (dec l)
         (let [;; retrieve relevant parameters
               dZ          (first (:dZ grads))      ; size (n[l],   m)
               W           (nth (:W net-params) l)  ; size (n[l],   n[l-1])
               A-prev      (nth (:A state) (dec l)) ; size (n[l-1], m)
               Z-prev      (nth (:Z state) (dec l)) ; size (n[l-1], m)
               fn-key-prev (nth (:fns net-params) (dec l))
               dfn-prev    (get-in activation-fns [fn-key-prev :dfn])

               ;; perform differential calculations
               dW (M// (m/dot
                        dZ
                        (m/transpose A-prev)) m)
               db (hnn-m/matrix-row-mean-keep-dims dZ)
               dA-prev (if (= l 1)
                         nil ; hack to avoid going back too far
                         (m/dot (m/transpose W) dZ))
               dZ-prev (if (= l 1)
                         nil ; hack to avoid going back too far
                         (M/* (m/emap dfn-prev Z-prev)
                              dA-prev))]

           ;; record gradients for this layer
           (assoc grads
                  :dW (conj (:dW grads) dW)
                  :db (conj (:db grads) db)
                  :dA (conj (:dA grads) dA-prev)
                  :dZ (conj (:dZ grads) dZ-prev))))))))

(defn train-neural-network
  [X Y net-params learning-rate num-iterations]
  (if (= 0 num-iterations)
    (do
      (let [A (last (:A (forward-prop X net-params)))
            final-cost (cost Y A)
            final-acc (accuracy Y A)]
        (println "FINAL cost:" final-cost
                 "FINAL accuary:" final-acc))
      net-params)
    (let [state (forward-prop X net-params)
          A (-> state :A last)
          cost (cost Y A)
          _ (when (= 0 (mod num-iterations 100))
              (println "cost:" cost num-iterations
                       "iterations to go, accuracy: " (accuracy Y A)))
          grads (back-prop X Y net-params state)
          new-net-params (update-net-params net-params
                                            grads
                                            learning-rate)]
      (recur X Y new-net-params learning-rate (dec num-iterations)))))

(defn predict
  "Given data X (shape n,m) and a network, predicts whether
  each of the m cases is true or false"
  [X net-params]
  (let [state (forward-prop X net-params)
        activations (-> state :A last)]
    (m/emap (partial < 0.5) activations)))
