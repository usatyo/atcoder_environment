/**
 * @file 01_nearest_neighbor_answer.cpp
 * @brief
 * 今いる点から最も近いレストランに行くことを50回繰り返し、その後今いる点から最も近い目的地に行くことを50回繰り返す解法プログラム
 */
#include <chrono>
#include <iostream>
#include <vector>

/// @brief 2次元座標上の点を表す構造体
struct Point {
    int x;
    int y;

    Point(int x, int y) : x(x), y(y) {}

    /// @brief 2点間のマンハッタン距離を計算する
    /// @param p 距離を計算する点
    /// @return 2点間のマンハッタン距離
    int dist(const Point& p) const { return std::abs(x - p.x) + std::abs(y - p.y); }
};

/// @brief 入力データを表す構造体
struct Input {
    /// @brief レストランの数 (=1000)
    int order_count;
    /// @brief 選択する必要のある注文の数 (=50)
    int pickup_count;
    /// @brief AtCoderオフィスの座標 (=(400, 400))
    Point office;
    /// @brief レストランの座標の配列
    std::vector<Point> restaurants;
    /// @brief 目的地の座標の配列
    std::vector<Point> destinations;

    /// @brief 入力データを読み込む
    /// @return 読み込んだ入力データ
    static Input read() {
        int order_count = 1000;
        int pickup_count = 50;
        Point office(400, 400);
        std::vector<Point> restaurants;
        std::vector<Point> destinations;

        for (int i = 0; i < order_count; ++i) {
            int a, b, c, d;
            std::cin >> a >> b >> c >> d;
            restaurants.emplace_back(a, b);
            destinations.emplace_back(c, d);
        }

        return Input{order_count, pickup_count, office, restaurants, destinations};
    }
};

/// @brief 出力データを表す構造体
struct Output {
    /// @brief 移動距離の合計
    int dist_sum;

    /// @brief 選択した注文のリスト
    std::vector<int> orders;

    /// @brief 配達ルート
    std::vector<Point> route;

    /// @brief 出力データを構築する
    /// @param orders 選択した注文のリスト
    /// @param route 配達ルート
    Output(std::vector<int> orders, std::vector<Point> route) : orders(orders), route(route) {
        // 移動距離の合計を計算する
        dist_sum = 0;

        for (int i = 0; i < route.size() - 1; ++i) {
            dist_sum += route[i].dist(route[i + 1]);
        }
    }

    /// @brief 解を出力する
    void print() {
        // 選択した注文の集合を出力する
        std::cout << orders.size();

        for (int i = 0; i < orders.size(); ++i) {
            // 0-indexed -> 1-indexedに変更
            std::cout << " " << orders[i] + 1;
        }

        std::cout << std::endl;

        // 配達ルートを出力する
        std::cout << route.size();

        for (int i = 0; i < route.size(); ++i) {
            std::cout << " " << route[i].x << " " << route[i].y;
        }

        std::cout << std::endl;
    }
};

/// @brief 問題を解く関数（この関数を実装していきます）
/// @param input 入力データ
/// @return 出力データ
Output solve(const Input& input) {
    // 貪欲その1
    // 以下を順に実行するプログラム
    // 1.高橋君は最初オフィスから出発する
    // 2.訪問したレストランが50軒に達するまで、今いる場所から一番近いレストランに移動することを繰り返す
    // 3.受けた注文を捌ききるまで、今いる場所から一番近い配達先に移動することを繰り返す
    // 4.オフィスに帰る

    std::vector<int> orders;   // 注文の集合
    std::vector<Point> route;  // 配達ルート

    // 1.オフィスからスタート
    route.push_back(input.office);
    Point current_position = input.office;  // 現在地
    int total_dist = 0;                     // 総移動距離

    // 2.訪問したレストランが50軒に達するまで、今いる場所から一番近いレストランに移動することを繰り返す

    // 同じレストランを2回訪れてはいけないので、訪問済みのレストランを記録する
    std::vector<bool> visited_restaurant(input.order_count, false);

    // pickup_count(=50)回ループ
    for (int i = 0; i < input.pickup_count; ++i) {
        // レストランを全探索して、最も近いレストランを探す
        int nearest_restaurant = 0;  // レストランの番号
        int min_dist = 1000000;      // 最も近いレストランの距離

        for (int j = 0; j < input.order_count; ++j) {
            // 【穴埋め】既に訪れていたらスキップ
            if (visited_restaurant[j]) {
                continue;
            }

            // 【穴埋め】最短距離が更新されたら記録
            // 【ヒント】int distance = p0.dist(p1); と書くと、p0とp1のマンハッタン距離が計算できる
            // 【ヒント】nearest_restaurant, min_distの2つを更新する
            int distance = current_position.dist(input.restaurants[j]);

            if (distance < min_dist) {
                min_dist = distance;
                nearest_restaurant = j;
            }
        }

        // 最も近いレストラン(nearest_restaurant)に移動する
        // 【穴埋め】現在位置を最も近いレストランの位置に更新
        current_position = input.restaurants[nearest_restaurant];

        // 【穴埋め】注文の集合に選んだレストランを追加
        orders.push_back(nearest_restaurant);

        // 【穴埋め】配達ルートに現在の位置を追加
        route.push_back(current_position);

        // 【穴埋め】訪問済みレストランの配列にtrueをセット
        visited_restaurant[nearest_restaurant] = true;

        // 総移動距離の更新
        total_dist += min_dist;

        // デバッグしやすいよう、標準エラー出力にレストランを出力
        // 標準エラー出力はデバッグに有効なので、AHCでは積極的に活用していきましょう
        Point restaurant_pos = input.restaurants[nearest_restaurant];
        std::cerr << i << "番目のレストラン: p_" << nearest_restaurant << " = (" << restaurant_pos.x << ", "
                  << restaurant_pos.y << ")" << std::endl;
    }

    // 【ヒント】ここまで穴埋めできたら、正しく動くか一度実行してみましょう！

    // 3.受けた注文を捌ききるまで、今いる場所から一番近い配達先に移動することを繰り返す

    // 行かなければいけない配達先のリスト
    // ordersは最終的に出力しなければならないので、ここでコピーを作成しておく
    // 配達先を訪問したらこのリストから1つずつ削除していく
    std::vector<int> destinations(orders);

    // pickup_count(=50)回ループ
    for (int i = 0; i < input.pickup_count; ++i) {
        // 配達先を全探索して、最も近い配達先を探す
        int nearest_index = 0;                                  // 配達先リストのインデックス
        int nearest_destination = destinations[nearest_index];  // 配達先の番号
        int min_dist = 1000000;                                 // 最も近い配達先の距離

        // 0～999まで全探索するのではなく、50個のレストランに対応した配達先を全探索することに注意
        for (int j = 0; j < destinations.size(); ++j) {
            // 【穴埋め】最短距離が更新されたら記録
            // 【ヒント】nearest_index, nearest_destination, min_distの3つを更新する
            int distance = current_position.dist(input.destinations[destinations[j]]);

            if (distance < min_dist) {
                min_dist = distance;
                nearest_index = j;
                nearest_destination = destinations[j];
            }
        }

        // 最も近い配達先(nearest_destination)に移動する
        // 【穴埋め】現在位置を最も近い配達先の位置に更新
        current_position = input.destinations[nearest_destination];

        // 【穴埋め】配達ルートに現在の位置を追加
        route.push_back(current_position);

        // 【穴埋め】配達先のリストから削除
        destinations.erase(destinations.begin() + nearest_index);

        // 総移動距離の更新
        total_dist += min_dist;

        // デバッグしやすいよう、標準エラー出力に配達先を出力
        Point destination_pos = input.destinations[nearest_destination];
        std::cerr << i << "番目の配達先: q_" << nearest_destination << " = (" << destination_pos.x << ", "
                  << destination_pos.y << ")" << std::endl;
    }

    // 4.オフィスに戻る
    route.push_back(input.office);
    total_dist += current_position.dist(input.office);

    // 合計距離を標準エラー出力に出力
    std::cerr << "total distance: " << total_dist << std::endl;

    return Output(orders, route);
}

int main() {
    // 入力データを読み込む
    Input input = Input::read();

    // 問題を解く
    Output output = solve(input);

    // 出力する
    output.print();

    return 0;
}