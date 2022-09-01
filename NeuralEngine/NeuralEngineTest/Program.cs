using NeuralEngine;

namespace NeuralEngineTest
{
    public class Programm
    {
        public class XORModelInput : IModelInput
        {
            public float x1 { get; set; }
            public float x2 { get; set; }
            public bool Class { get; set; }
        }
        public class XORModelOutput : IModelOutput
        {
            public bool Prediction { get; set; }
            public float Score { get; set; }
        }
        public static void Main()
        {
            var neu = new NeuralEngine<XORModelInput, XORModelOutput>();
            neu.Train(new XORModelInput[]
            {
                new XORModelInput(){ x1=0,x2=0,Class=false},
                new XORModelInput(){ x1=1,x2=0,Class=true},
                new XORModelInput(){ x1=0,x2=1,Class=true},
                new XORModelInput(){ x1=1,x2=1,Class=false},
            });

            var answ = neu.Think(new XORModelInput() { x1 = 0, x2 = 0 });
            answ = neu.Think(new XORModelInput() { x1 = 0, x2 = 1 });
            answ = neu.Think(new XORModelInput() { x1 = 1, x2 = 0 });
            answ = neu.Think(new XORModelInput() { x1 = 1, x2 = 1 });


            Console.ReadKey();
        }
    }
}
