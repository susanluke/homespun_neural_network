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
  (let [{:keys [W b]} (sut/init-net-params [2 3 1]
                                           (repeat 3 :identity))]
    (is (= [3 2] (m/shape (nth W 1))))
    (is (= [1 3] (m/shape (nth W 2))))
    (is (= 0 (nth b 1)))
    (is (= 0 (nth b 2)))))

(deftest update-net-params-test
  (let [net-params {:layer-sizes [2 3 2]
                    :W [nil
                        [[1 2 3]
                         [4 5 6]]
                        [[100 200]
                         [300 400]
                         [500 600]]]
                    :b [nil
                        [[1]
                         [2]
                         [3]]
                        [[4]
                         [5]]]}
        grads {:dW [nil
                    [[-10 -20 -30]
                     [40 50 60]]
                    [[-1 -2]
                     [3 4]
                     [5 6]]]
               :db [nil
                    [[-10]
                     [20]
                     [-30]]
                    [[40]
                     [-50]]]}
        learning-rate 0.5
        {:keys [W b]} (sut/update-net-params net-params
                                             grads
                                             learning-rate)]
    (is (= [[6.0 12.0 18.0]
            [-16.0 -20.0 -24.0]]
           (nth W 1)))
    (is (= [[100.5 201.0]
            [298.5 398.0]
            [497.5 597.0]]
           (nth W 2)))
    (is (= [[6.0]
            [-8.0]
            [18.0]]
           (nth b 1)))
    (is (= [[-16.0]
            [30.0]]
           (nth b 2)))))

(deftest forward-prop-test
  (testing "basic forward prop"
    (let [X [[1] [2]]
          net-params {:layer-sizes [2 3 2]
                      :W [nil
                          [[1 2]
                           [4 5]
                           [6 7]]
                          [[100 200 350]
                           [300 400 450]]]
                      :b [nil
                          [[1]
                           [2]
                           [3]]
                          [[4]
                           [5]]]
                      :fns (repeat 3 :identity)}
          state (sut/forward-prop X net-params)]
      (is (= (:A state)
             [X
              [[6] [16] [23]]
              [[11854] [18555]]]))
      (is (= (:Z state)
             [nil
              [[6] [16] [23]]
              [[11854] [18555]]])))))

;; TODO these 2 fns need refactoring based on new net-param format
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
