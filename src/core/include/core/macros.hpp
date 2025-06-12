//
// Created by VanHoai on 30/5/25.
//

#ifndef MACROS_HPP
#define MACROS_HPP

#include <vector>

#define ms(s, n)      memset(s, n, sizeof(s))
#define all(a)        a.begin(), (a).end()
#define sz(a)         int((a).size())
#define FOR(i, a, b)  for (int(i) = (a); (i) <= (b); ++(i))
#define FORD(i, a, b) for (int(i) = (a); (i) >= (b); --(i))

#define PB push_back
#define MP make_pair
#define MU make_unique
#define F  first
#define S  second

typedef long long ll;
typedef std::pair<int, int> pi;
typedef std::vector<int> vi;
typedef std::vector<pi> vii;
typedef std::vector<vi> vvi;

constexpr int MOD = static_cast<int>(1e9) + 7;
constexpr int INF = static_cast<int>(1e9) + 1;

#endif   // MACROS_HPP
