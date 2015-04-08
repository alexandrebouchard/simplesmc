package simplesmc;

import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.Test;

import simplesmc.hmm.HMMProblemSpecification;
import simplesmc.hmm.HMMUtils;
import simplesmc.hmm.ToyHMMParams;



public class TestSMC
{
  @Test
  public void testSMC()
  {
    Random random = new Random(1);
    ToyHMMParams hmmParams = new ToyHMMParams(5);
    
    Pair<List<Integer>, List<Integer>> generated = HMMUtils.generate(random, hmmParams, 10);
    List<Integer> observations = generated.getRight();
    
    System.out.println("exact = " + HMMUtils.exactDataLogProbability(hmmParams, observations));
    
    HMMProblemSpecification proposal = new HMMProblemSpecification(hmmParams, observations);
    
    SMCOptions options = new SMCOptions();
    SMCAlgorithm<Integer> smc = new SMCAlgorithm<>(proposal, options);
    System.out.println("estimate = " + smc.sample().logNormEstimate());
  }
}
