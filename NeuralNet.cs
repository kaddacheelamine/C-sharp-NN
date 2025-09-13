using System;
using System.Collections.Generic;
using System.Linq;

namespace NN
{
    public enum Act { Sig, ReLU }

    public static class MU
    {
        private static Random r = new Random();

        public static double RndN(double m = 0, double s = 1)
        {
            double u1 = 1.0 - r.NextDouble();
            double u2 = 1.0 - r.NextDouble();
            double n = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return m + s * n;
        }

        public static double Sig(double x) => 1.0 / (1.0 + Math.Exp(-x));
        public static double SigP(double x) { var s = Sig(x); return s * (1 - s); }

        public static double ReLU(double x) => Math.Max(0, x);
        public static double ReLUP(double x) => x > 0 ? 1 : 0;
    }

    public class NNets
    {
        public int[] sz;
        int L => sz.Length;
        double[][][] w;
        double[][] b;
        Act act;

        public NNets(int[] s, Act a = Act.Sig)
        {
            sz = s;
            act = a;
            Init();
        }

        void Init()
        {
            w = new double[L - 1][][];
            b = new double[L - 1][];

            for (int l = 0; l < L - 1; l++)
            {
                int nIn = sz[l], nOut = sz[l + 1];
                b[l] = new double[nOut];
                w[l] = new double[nOut][];

                double std = act == Act.Sig ? Math.Sqrt(1.0 / nIn) : Math.Sqrt(2.0 / nIn);

                for (int i = 0; i < nOut; i++)
                {
                    b[l][i] = MU.RndN(0, std);
                    w[l][i] = new double[nIn];
                    for (int j = 0; j < nIn; j++)
                        w[l][i][j] = MU.RndN(0, std);
                }
            }
        }

        public double[] FF(double[] x, out double[][] zs, out double[][] aa)
        {
            aa = new double[L][];
            zs = new double[L - 1][];
            aa[0] = x;

            double[] cur = x;
            for (int l = 0; l < L - 1; l++)
            {
                int nOut = sz[l + 1];
                double[] z = new double[nOut];
                double[] a = new double[nOut];

                for (int i = 0; i < nOut; i++)
                {
                    double sum = b[l][i];
                    var wr = w[l][i];
                    for (int j = 0; j < wr.Length; j++) sum += wr[j] * cur[j];
                    z[i] = sum;
                    a[i] = act == Act.Sig ? MU.Sig(sum) : MU.ReLU(sum);
                }
                zs[l] = z;
                aa[l + 1] = a;
                cur = a;
            }
            return cur;
        }

        public void Train(List<(double[] x, double[] y)> data, int ep, int mb, double lr)
        {
            var rnd = new Random();
            for (int e = 0; e < ep; e++)
            {
                data = data.OrderBy(_ => rnd.Next()).ToList();
                for (int k = 0; k < data.Count; k += mb)
                {
                    var batch = data.GetRange(k, Math.Min(mb, data.Count - k));
                    Upd(batch, lr);
                }
            }
        }

        void Upd(List<(double[] x, double[] y)> batch, double lr)
        {
            var nb = b.Select(bb => new double[bb.Length]).ToArray();
            var nw = w.Select(wl => wl.Select(row => new double[row.Length]).ToArray()).ToArray();

            foreach (var (x, y) in batch)
            {
                var (db, dw) = BP(x, y);
                for (int l = 0; l < nb.Length; l++)
                {
                    for (int i = 0; i < nb[l].Length; i++)
                        nb[l][i] += db[l][i];
                    for (int i = 0; i < nw[l].Length; i++)
                        for (int j = 0; j < nw[l][i].Length; j++)
                            nw[l][i][j] += dw[l][i][j];
                }
            }

            double sc = lr / batch.Count;
            for (int l = 0; l < w.Length; l++)
            {
                for (int i = 0; i < w[l].Length; i++)
                {
                    b[l][i] -= sc * nb[l][i];
                    for (int j = 0; j < w[l][i].Length; j++)
                        w[l][i][j] -= sc * nw[l][i][j];
                }
            }
        }

        (double[][], double[][][]) BP(double[] x, double[] y)
        {
            var nb = b.Select(bb => new double[bb.Length]).ToArray();
            var nw = w.Select(wl => wl.Select(row => new double[row.Length]).ToArray()).ToArray();

            var outp = FF(x, out var zs, out var aa);

            int Lw = L - 1;
            double[] d = new double[sz.Last()];
            var zL = zs[Lw - 1];
            var aL = aa.Last();
            for (int i = 0; i < d.Length; i++)
            {
                double cd = aL[i] - y[i];
                double ap = act == Act.Sig ? MU.SigP(zL[i]) : MU.ReLUP(zL[i]);
                d[i] = cd * ap;
            }
            nb[Lw - 1] = d.ToArray();
            var pa = aa[Lw - 1];
            for (int i = 0; i < nw[Lw - 1].Length; i++)
                for (int j = 0; j < nw[Lw - 1][i].Length; j++)
                    nw[Lw - 1][i][j] = d[i] * pa[j];

            for (int l = Lw - 2; l >= 0; l--)
            {
                int n = sz[l + 1];
                double[] z = zs[l];
                double[] sp = new double[n];
                for (int i = 0; i < n; i++)
                    sp[i] = act == Act.Sig ? MU.SigP(z[i]) : MU.ReLUP(z[i]);

                double[] dNew = new double[n];
                for (int i = 0; i < n; i++)
                {
                    double sum = 0;
                    for (int k = 0; k < w[l + 1].Length; k++)
                        sum += w[l + 1][k][i] * d[k];
                    dNew[i] = sum * sp[i];
                }
                d = dNew;

                nb[l] = d.ToArray();
                pa = aa[l];
                for (int i = 0; i < nw[l].Length; i++)
                    for (int j = 0; j < nw[l][i].Length; j++)
                        nw[l][i][j] = d[i] * pa[j];
            }
            return (nb, nw);
        }

        public double[] Pred(double[] x) => FF(x, out _, out _);
    }

    class P
    {
        static void Main(string[] args)
        {
            // توليد البيانات: 5 مدخلات، 32 عينة
            var data = new List<(double[] x, double[] y)>();
            for (int a = 0; a <= 1; a++)
            for (int b = 0; b <= 1; b++)
            for (int c = 0; c <= 1; c++)
            for (int d = 0; d <= 1; d++)
            for (int e = 0; e <= 1; e++)
            for (int f = 0; f <= 1; f++)
            {
                int y = ((a | b) ^ ((c ^ f) & (d | e)));
                data.Add((new double[]{a,b,c,d,e}, new double[]{y}));
            }

            // تقسيم: 22 للتدريب، 10 للاختبار
            var rnd = new Random();
            data = data.OrderBy(_ => rnd.Next()).ToList();
            var train = data.Take(54).ToList();
            var test = data.Skip(54).ToList();

            var net = new NNets(new int[]{5, 8, 1}, Act.Sig);

            Console.WriteLine("Training...");
            net.Train(train, ep: 5000, mb: 4, lr: 0.002);

            int correct = 0;
            foreach (var (x,y) in test)
            {
                var p = net.Pred(x)[0];
                int pred = p > 0.5 ? 1 : 0;
                if (pred == (int)y[0]) correct++;
                Console.WriteLine($"[{string.Join(",",x)}] => {p:F3} -> {pred} (target {y[0]})");
            }
            double acc = (double)correct / test.Count;
            Console.WriteLine($"\nACC on test = {acc:P2}");
        }
    }
}
