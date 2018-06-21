(defproject homespun_neural_network "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [net.mikera/core.matrix "0.62.0"]
                 [com.gfredericks/test.chuck "0.2.9"]
                 [org.clojure/test.check "0.9.0"]]
  :main ^:skip-aot homespun-neural-network.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
