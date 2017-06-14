package simplesmc.hmm;

import bayonet.distributions.Random; 

/**
 * A simple, "sticky" HMM, where the latent variable at each
 * step stays at the same location with probability given by 
 * selfTransitionProbability, otherwise jump to another state 
 * uniformly at random.
 * 
 * Similarly, the emissions are identical to the latent state 
 * with probability 1-noiseProbability, and otherwise emits 
 * one of the other states with equal probability.
 * 
 * @author Alexandre Bouchard (alexandre.bouchard@gmail.com)
 *
 */
public class ToyHMMParams implements HMMParams
{
  public final double selfTransitionProbability = 0.9;
  
  public static final double noiseProbability = 0.1;
  public final int nStates;

  @Override
  public double initialLogPr(int state)
  {
    return Math.log(1.0/nStates);
  }

  @Override
  public int sampleInitial(Random random)
  {
    return random.nextInt(nStates);
  }

  @Override
  public double transitionLogPr(int currentState, int nextState)
  {
    return logPr(currentState, nextState, selfTransitionProbability);
  }
  
  private double logPr(int currentState, int nextState, double selfTransitionProbability)
  {
    if (currentState == nextState)
      return Math.log(selfTransitionProbability);
    else
      return Math.log((1.0 - selfTransitionProbability) / (nStates - 1));
  }

  @Override
  public int sampleTransition(Random random, int currentState)
  {
    return sample(random, currentState, selfTransitionProbability);
  }
  
  private int sample(Random random, int currentState, double selfTransitionProbability)
  {
    if (random.nextBernoulli(selfTransitionProbability))
      return currentState;
    else
    {
      int randomIndex = random.nextInt(nStates - 1);
      if (randomIndex < currentState)
        return randomIndex;
      else
        return randomIndex + 1;
    }
  }

  @Override
  public double emissionLogPr(int latentState, int emission)
  {
    return logPr(latentState, emission, 1.0 - noiseProbability);
  }

  @Override
  public int nLatentStates()
  {
    return nStates;
  }

  @Override
  public int nObservedStates()
  {
    return nStates;
  }

  @Override
  public int sampleEmission(Random random, int currentState)
  {
    return sample(random, currentState, 1.0 - noiseProbability);
  }

  public ToyHMMParams(int nStates)
  {
    this.nStates = nStates;
  }
}
