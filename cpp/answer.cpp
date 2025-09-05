#include <bits/stdc++.h>
using namespace std;
#include <atcoder/all>
using namespace atcoder;
using ll = long long;
using pii = pair<int, int>;
using pll = pair<ll, ll>;
using vi = vector<int>;
using vvi = vector<vi>;
using vl = vector<ll>;
using vvl = vector<vl>;
using vs = vector<string>;

#define rep(i, x, limit) for (int i = (int)x; i < (int)limit; i++)
#define el '\n'
#define Yes cout << "Yes" << el
#define No cout << "No" << el
#define eps (1e-10)
#define Equals(a, b) (fabs((a) - (b)) < eps)

const double pi = 3.141592653589793238;
const int inf = 1073741823;
const ll infl = 1LL << 60;
const string ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const string abc = "abcdefghijklmnopqrstuvwxyz";

ll powi(ll a, ll b) {
  ll ret = 1;
  rep(i, 0, b) { ret *= a; }
  return ret;
}

using mint = atcoder::modint998244353;

const int MAX = 5100000;
mint fac[MAX], finv[MAX], inv[MAX];

// テーブルを作る前処理
void combInit() {
  const int MOD = mint::mod();
  fac[0] = fac[1] = 1;
  finv[0] = finv[1] = 1;
  inv[1] = 1;
  for (int i = 2; i < MAX; i++) {
    fac[i] = fac[i - 1] * i;
    inv[i] = MOD - inv[MOD % i] * (MOD / i);
    finv[i] = finv[i - 1] * inv[i];
  }
}

// 二項係数計算
mint comb(int n, int k) {
  if (n < k)
    return 0;
  if (n < 0 || k < 0)
    return 0;
  return fac[n] * finv[k] * finv[n - k];
}

int main() {
  cout << 1 << endl;
  return 0;
}
