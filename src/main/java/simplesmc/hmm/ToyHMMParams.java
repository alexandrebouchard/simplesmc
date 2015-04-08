package simplesmc.hmm;

import java.util.Random;

import bayonet.distributions.Multinomial;



public class ToyHMMParams implements HMMParams
{
  private final double [][] transitionPrs = new double[][]{{0.8,0.15,0.05},{0.0,0.95,0.05},{0.15,0.15,0.7}};
  private final double [][] emissionPrs = new double[][]{{0.8,0.15,0.04,0.01},{0.0,0.95,0.04,0.01},{0.15,0.15,0.6,0.1}};
  private final double [] initialPrs = new double[]{0.25, 0.25, 0.5};

  @Override
  public double initialLogPr(int state)
  {
    return Math.log(initialPrs[state]);
  }

  @Override
  public int sampleInitial(Random random)
  {
    return Multinomial.sampleMultinomial(random, initialPrs);
  }

  @Override
  public double transitionLogPr(int currentState, int nextState)
  {
    return Math.log(transitionPrs[currentState][nextState]);
  }

  @Override
  public int sampleTransition(Random random, int currentState)
  {
    return Multinomial.sampleMultinomial(random, transitionPrs[currentState]);
  }

  @Override
  public double emissionLogPr(int latentState, int emission)
  {
    return Math.log(emissionPrs[latentState][emission]);
  }

  @Override
  public int nLatentStates()
  {
    return transitionPrs.length;
  }

  @Override
  public int nPossibleObservations()
  {
    return emissionPrs[0].length;
  }

  @Override
  public int sampleEmission(Random random, int currentState)
  {
    return Multinomial.sampleMultinomial(random, emissionPrs[currentState]);
  }

}
