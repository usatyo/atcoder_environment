// #pragma GCC target("avx2")
// #pragma GCC optimize("O3")
// #pragma GCC optimize("unroll-loops")
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, m;
  cin >> n >> m;
  vector<int> A(m);
  for (int i = 0; i < m; i++) {
    cin >> A[i];
  }
  if (accumulate(A.begin(), A.end(), 0) != n * (n + 1) / 2) {
    cout << "Input Error" << endl;
    return 0;
  }
  map<tuple<int, vector<int>, vector<int>>, bool> memo;
  vector<bool> shown(m + 1, false);
  auto F = [&](auto &&F, int x, vector<int> color, vector<int> B) -> bool {
    if (x == n) {
      bool res = true;
      for (int i = 0; i < m; i++) {
        res &= B[i] == 0;
      }
      if (res && !shown[x]) {
        for (int i = 0; i < x; i++) {
          cout << color[i] << (i == x - 1 ? '\n' : ' ');
        }
        shown[x] = true;
      }
      return res;
    }
    auto T = make_tuple(x, color, B);
    if (memo.count(T)) {
      return memo[T];
    }
    bool check = true;
    for (int i = 0; i < m; i++) {
      check &= B[i] > 0;
    }
    if (!check) {
      memo[T] = false;
      return false;
    }
    bool ret = false;
    for (int bit = 0; bit < 1 << x; bit++) {
      vector<int> next_color(x + 1, -1);
      bool ok = true;
      for (int i = 0; i < x; i++) {
        if (bit >> i & 1) {
          if (next_color[i] == color[i] || next_color[i] == -1) {
            next_color[i] = color[i];
          } else {
            ok = false;
          }
        } else {
          if (next_color[i + 1] == color[i] || next_color[i + 1] == -1) {
            next_color[i + 1] = color[i];
          } else {
            ok = false;
          }
        }
      }
      if (!ok) {
        continue;
      }
      for (int i = 0; i <= x; i++) {
        if (next_color[i] >= 0) {
          B[next_color[i]]--;
        }
      }
      vector<int> undefined;
      for (int i = 0; i <= x; i++) {
        if (next_color[i] == -1) {
          undefined.push_back(i);
        }
      }
      int siz = undefined.size();
      auto G = [&](auto &&G, int now) {
        if (now == siz) {
          ret |= F(F, x + 1, next_color, B);
          return;
        }
        for (int i = 0; i < m; i++) {
          if (B[i] > 0) {
            next_color[undefined[now]] = i;
            B[i]--;
            G(G, now + 1);
            B[i]++;
            next_color[undefined[now]] = -1;
          }
        }
      };
      G(G, 0);
      for (int i = 0; i <= x; i++) {
        if (next_color[i] >= 0) {
          B[next_color[i]]++;
        }
      }
    }
    if (ret && !shown[x]) {
      for (int i = 0; i < n - x; i++) {
        cout << ' ';
      }
      for (int i = 0; i < x; i++) {
        cout << color[i] << (i == x - 1 ? '\n' : ' ');
      }
      shown[x] = true;
    }
    memo[T] = ret;
    return ret;
  };
  for (int i = 0; i < m; i++) {
    A[i]--;
    if (F(F, 1, {i}, A)) {
      cout << "Yes" << endl;
      return 0;
    }
    A[i]++;
  }
  cout << "No" << endl;
}
