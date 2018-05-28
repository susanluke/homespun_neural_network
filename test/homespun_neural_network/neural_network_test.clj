(ns homespun-neural-network.neural-network-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [homespun-neural-network.neural-network :as sut]))

(deftest relu-test
  (are [in out] (= (sut/relu in) out)
    -10 0
    -1  0
    0   0
    1   1
    10  10))

(deftest relu-grad
  (are [in out] (= (sut/relu-grad in) out)
    -10 0
    -1  0
    0   0
    1   1
    10  1))

(deftest init-net-params-test
  (let [{:keys [W b]} (sut/init-net-params [2 3 1])]
    (is (= [3 2] (m/shape (nth W 1))))
    (is (= [1 3] (m/shape (nth W 2))))
    (is (= 0 (nth b 1)))
    (is (= 0 (nth b 2)))))

(deftest update-net-params-test
  (let [net-params {:layer-sizes [2 3 2]
                    :W1 [[1 2 3]
                         [4 5 6]]
                    :W2 [[100 200]
                         [300 400]
                         [500 600]]
                    :b1 [[1]
                         [2]
                         [3]]
                    :b2 [[4]
                         [5]]}
        grads {:dW1 [[-10 -20 -30]
                     [40 50 60]]
               :dW2 [[-1 -2]
                     [3 4]
                     [5 6]]
               :db1 [[-10]
                     [20]
                     [-30]]
               :db2 [[40]
                     [-50]]}
        learning-rate 0.5
        new-net-params (sut/update-net-params net-params grads learning-rate)]
    (is (= [[6.0 12.0 18.0]
            [-16.0 -20.0 -24.0]]
           (:W1 new-net-params)))
    (is (= [[100.5 201.0]
            [298.5 398.0]
            [497.5 597.0]]
           (:W2 new-net-params)))
    (is (= [[6.0]
            [-8.0]
            [18.0]]
           (:b1 new-net-params)))
    (is (= [[-16.0]
            [30.0]]
           (:b2 new-net-params)))))

(defn net-params->vector
  [net-params]
  (let [num-layers (sut/num-layers net-params)]
    (loop [layer 1
           v []]
      (if (= num-layers layer)
        v
        (let [W ((sut/make-key :W layer) net-params)
              b ((sut/make-key :b layer) net-params)
              W-size (apply * (m/shape W))
              b-size (apply * (m/shape b))]
          (recur (inc layer)
                 (concat v
                         (m/reshape W [W-size])
                         (m/reshape b [b-size]))))))))

(defn vector->net-params
  [v layer-sizes]
  (let [num-layers (count layer-sizes)]
    (loop [net-params {:layer-sizes layer-sizes}
           layer 1
           v v]
      (if (= num-layers layer)
        net-params
        (let [num-nodes (nth layer-sizes layer)
              num-nodes-prev (nth layer-sizes (dec layer))
              W-size [num-nodes num-nodes-prev]
              b-size [num-nodes 1]
              W-num (apply * W-size)]
          (recur (assoc net-params
                        (sut/make-key :W layer)
                        (m/reshape v W-size)
                        (sut/make-key :b layer)
                        (m/reshape (drop W-num v) b-size))
                 (inc layer)
                 (drop (+ W-num num-nodes) v)))))))



;; TODO need to check the functions above and implement grad-check
