package simplesmc.hmm;

import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.tuple.Pair;

import simplesmc.SMCProposal;
import simplesmc.pmcmc.WithSignature;



public class HMMProposal implements SMCProposal<Integer>, WithSignature
{
  private final HMMParams parameters;
  private final List<Integer> observations;
  
  @Override
  public Pair<Double, Integer> proposeNext(int previousSmcIteration,
      Random random, Integer currentParticle)
  {
    int proposed = parameters.sampleTransition(random, currentParticle);
    double emissionLogPr = parameters.emissionLogPr(proposed, observations.get(previousSmcIteration + 1));
    return Pair.of(emissionLogPr, proposed);
  }
  
  @Override
  public Pair<Double, Integer> proposeInitial(Random random)
  {
    int proposed = parameters.sampleInitial(random);
    double emissionLogPr = parameters.emissionLogPr(proposed, observations.get(0));
    return Pair.of(emissionLogPr, proposed);
  }

  public HMMProposal(HMMParams parameters, List<Integer> observations)
  {
    this.parameters = parameters;
    this.observations = observations;
  }

  @Override
  public int nIterations()
  {
    return observations.size();
  }

  @Override
  public long signature()
  {
    return parameters.signature();
  }

}
